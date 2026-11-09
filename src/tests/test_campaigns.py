"""CPU-only server-wrapper smoke tests; no models or dataset assets are used."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from tools import campaign_status as status
from tools import select_sweep_lr
from src.training.recovery import fingerprint

ROOT = Path(__file__).resolve().parents[2]


def clean_env(**overrides):
    env = dict(os.environ)
    for key in ('RUN_P0', 'RUN_P1', 'RUN_P2', 'SEEDS', 'ONLY', 'SKIP', 'ICL_14B', 'RUN_14B_LORA', 'RUN_GALORE'):
        env.pop(key, None)
    return dict(env, DRY='1', RCG_PYTHON=sys.executable, **overrides)


def test_six_server_wrappers_preview_without_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', clean_env())
    scripts = sorted((ROOT / 'scripts/slurm').glob('*.sbatch'))
    assert {p.stem for p in scripts} == {'00_07_infrastructure', *status.SCRIPTS.values(), '16_campaign_j'}
    for path in scripts:
        text = path.read_text()
        assert '#SBATCH --time=72:00:00' in text
        assert '#SBATCH --signal=B:USR1@600' in text
        assert '#SBATCH --gres=gpu:1' in text
        subprocess.run(['bash', '-n', str(path)], check=True)
        subprocess.run(['bash', str(path)], env=clean_env(RCG_RESULTS_ROOT=str(tmp_path/'results'),
                        RCG_LOG_ROOT=str(tmp_path/'logs'), RCG_PREFLIGHT_ROOT=str(tmp_path/'probes')),
                       cwd=ROOT, capture_output=True, check=True)
    assert list(tmp_path.iterdir()) == []
    assert {phase: len(status.commands(phase, tmp_path)) for phase in status.SCRIPTS} == dict(A=28, B=5, C=66, D=4)


def test_missing_campaign_cannot_be_complete(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('SEEDS', '42')
    monkeypatch.setenv('RUN_P1', '0')
    assert status.main(['A', '--results-root', str(tmp_path)]) == 1
    assert 'incomplete' in capsys.readouterr().out


@pytest.mark.parametrize('root_variable', ['SLURM_SUBMIT_DIR', 'RCG_ROOT'])
def test_spooled_scripts_resolve_repository_helpers(tmp_path, root_variable):
    """Slurm copies only the sbatch file, not its sibling common.sh."""
    for script in (ROOT / 'scripts/slurm').glob('*.sbatch'):
        spool = tmp_path / 'slurm_script'
        spool.write_text(script.read_text())
        env = clean_env()
        env.pop('RCG_ROOT', None)
        env.pop('SLURM_SUBMIT_DIR', None)
        env[root_variable] = str(ROOT)
        subprocess.run(['bash', str(spool)], cwd=tmp_path, env=env,
                       capture_output=True, check=True)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['slurm_script']


def test_profile_seed42_uses_default_name(tmp_path):
    command = [sys.executable, '-m', 'src.train', '--model', 'codet5', '--method', 'lora',
               '--memory-profile', 'a100', '--results-root', str(tmp_path), '--dry-run']
    default = json.loads(subprocess.check_output(command, cwd=ROOT))
    explicit = json.loads(subprocess.check_output(command + ['--seed', '42'], cwd=ROOT))
    assert default == explicit


def test_per_phase_scope_can_check_all_supported_optional_work(tmp_path, monkeypatch):
    monkeypatch.setattr(os, 'environ', clean_env())
    monkeypatch.setenv('RCG_A_RUN_P1', '1')
    monkeypatch.setenv('RCG_C_RUN_P2', '1')
    assert {phase: len(status.commands(phase, tmp_path)) for phase in status.SCRIPTS} == dict(A=38, B=5, C=78, D=4)


def completed_sweep(root, index, metric):
    run = root / f'codet5_lora_{index}'
    checkpoint = run / 'checkpoint-1'; checkpoint.mkdir(parents=True)
    identity = {'fixture': index}; stamp = fingerprint(identity)
    (run/'run_identity.json').write_text(json.dumps(dict(identity=identity, fingerprint=stamp)))
    result = {k: 0 for k in ['test_bleu_4', 'test_exact_match', 'test_rouge', 'test_bertscore', 'test_meteor']}
    result.update(run_fingerprint=stamp, expected_test_rows=1, raw_data=[dict(prediction='x', reference='y')])
    (run/'complete_results.json').write_text(json.dumps(result))
    for name in ['optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'adapter_model.safetensors']:
        (checkpoint/name).write_text('fixture')
    (checkpoint/'trainer_state.json').write_text(json.dumps(dict(global_step=1, best_metric=metric)))
    files = {p.name: p.stat().st_size for p in checkpoint.iterdir()}
    (checkpoint/'recovery_complete.json').write_text(json.dumps(dict(global_step=1, files=files)))
    return run


def test_sweep_requires_full_bracket_and_committed_state(tmp_path, capsys):
    command = ['--sweep-root', str(tmp_path), '--stem', 'codet5_lora', '--quiet', '--require-complete']
    completed_sweep(tmp_path, 0, 10)
    completed_sweep(tmp_path, 1, 20)
    assert select_sweep_lr.main(command) == 1
    run = completed_sweep(tmp_path, 2, 15)
    assert select_sweep_lr.main(command) == 0
    assert capsys.readouterr().out.strip() == str(select_sweep_lr.LR[1])
    (run/'checkpoint-1'/'optimizer.pt').write_text('torn')
    assert select_sweep_lr.main(command) == 1


def shell_worker(tmp_path, body):
    script = tmp_path/'wrapper.sh'
    script.write_text('source scripts/slurm/common.sh\nDRY=0\nLOG_ROOT="$1"\n' + body)
    return subprocess.Popen(['bash', str(script), str(tmp_path)], cwd=ROOT, env=clean_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_failure_stops_dependents(tmp_path):
    proc = shell_worker(tmp_path, 'run_step failing bash -c "exit 7"\nrun_step dependent touch "$1/dependent"\n')
    proc.communicate(timeout=15)
    assert proc.returncode == 7
    assert not (tmp_path/'dependent').exists()


def test_signal_waits_for_checkpoint_then_stops(tmp_path):
    worker = tmp_path/'worker.py'
    worker.write_text('''import pathlib, signal, sys, time
root = pathlib.Path(sys.argv[1])
def stop(*args):
    time.sleep(0.15)
    (root/'saved').write_text('committed')
    sys.exit(75)
signal.signal(signal.SIGUSR1, stop)
(root/'ready').touch()
while True: time.sleep(0.05)
''')
    proc = shell_worker(tmp_path, f'run_step worker "{sys.executable}" "$1/worker.py" "$1"\nrun_step dependent touch "$1/dependent"\n')
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path/'ready').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert (tmp_path/'ready').exists()
        proc.send_signal(signal.SIGUSR1)
        proc.communicate(timeout=15)
        assert proc.returncode == 75
        assert (tmp_path/'saved').exists()
        assert not (tmp_path/'dependent').exists()
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait()


def test_filtered_training_does_not_require_admission(tmp_path):
    proc = shell_worker(tmp_path, 'ONLY="included"\ntrain_official excluded trans_review codet5 lora 5e-5\nfinish_report\n')
    output, _ = proc.communicate(timeout=15)
    assert proc.returncode == 75
    assert 'SKIP excluded' in output


def test_campaign_gate_accepts_bound_complete_artifacts_then_rejects_changes(tmp_path, monkeypatch):
    from src.train import build_parser, recovery_identity, resolve_protocol_overrides
    from src.training.registry import get_spec, resolve_run
    from src.training.memory_profile import apply_memory_profile
    data = tmp_path/'data.jsonl'; data.write_text('{"fixture": 1}\n')
    monkeypatch.setitem(status.DATASET_CONFIGS, 'trans_review', {'data_files': dict(train=str(data), val=str(data), test=str(data))})
    argv = [sys.executable, '-m', 'src.train', '--model', 'codet5', '--method', 'lora',
            '--learning_rate', '5e-5', '--dataset', 'trans_review', '--results-root', str(tmp_path),
            '--memory-profile', 'a100', '--gen-batch-size', '32', '--attn-implementation', 'sdpa']
    args = build_parser().parse_args(argv[3:])
    model, method, spec = get_spec(args.model, args.method)
    spec, overrides = apply_memory_profile(spec, resolve_protocol_overrides(model, args), 'a100', model, method)
    run = resolve_run(spec, learning_rate=args.learning_rate, results_root=args.results_root)
    root = Path(run['experiment_results_dir']); root.mkdir()
    identity = recovery_identity(spec, overrides, args, run)
    stamp = fingerprint(identity)
    (root/'run_identity.json').write_text(json.dumps(dict(identity=identity, fingerprint=stamp)))
    result = {k: 0 for k in ['test_bleu_4', 'test_exact_match', 'test_rouge', 'test_bertscore', 'test_meteor']}
    result.update(run_fingerprint=stamp, expected_test_rows=1, raw_data=[dict(prediction='x', reference='y')])
    (root/'complete_results.json').write_text(json.dumps(result))
    icl_argv = [sys.executable, '-m', 'src.icl_inference', '--model_name', 'qwen25',
                '--dataset', 'trans_review', '--num_examples', '0', '--batch-size', '8',
                '--results_dir', str(tmp_path)]
    path = tmp_path/'qwen25_icl_0ex_random_v2.json'
    icl_identity = dict(model_name='qwen25', dataset='trans_review', num_examples=0,
                        selection_method='random', demo_seed=None, prompt_version='v2',
                        memory_profile='a100', batch_size=8, max_prompt_tokens=2048,
                        attn_implementation='sdpa', seed=42, allow_exact_match=False,
                        max_samples=None, llm_judge=False, sample_index_file=None,
                        train_data_sha256=status.digest(str(data)), test_data_sha256=status.digest(str(data)))
    icl_stamp = fingerprint(icl_identity)
    Path(str(path)+'.identity.json').write_text(json.dumps(dict(identity=icl_identity, fingerprint=icl_stamp)))
    icl = {k: 0 for k in ['bleu_score', 'exact_match_score', 'rouge_score', 'bertscore', 'meteor_score']}
    icl.update(run_fingerprint=icl_stamp, num_test_samples=1, predictions=['x'], references=['y'], prompts=['p'])
    path.write_text(json.dumps(icl))
    monkeypatch.setattr(status, 'commands', lambda *args: [dict(label='train', argv=argv), dict(label='icl', argv=icl_argv)])
    assert status.main(['A', '--results-root', str(tmp_path)]) == 0
    icl['num_test_samples'] = 2; path.write_text(json.dumps(icl))
    assert status.main(['A', '--results-root', str(tmp_path)]) == 1
    icl['num_test_samples'] = 1; path.write_text(json.dumps(icl))
    identity['seed'] = 3
    (root/'run_identity.json').write_text(json.dumps(dict(identity=identity, fingerprint=fingerprint(identity))))
    assert status.main(['A', '--results-root', str(tmp_path)]) == 1


def test_campaign_gate_validates_sweep_subset_coverage(tmp_path, monkeypatch):
    from src.train import build_parser, recovery_identity, resolve_protocol_overrides
    from src.training.registry import get_spec, resolve_run
    from src.training.memory_profile import apply_memory_profile
    data = tmp_path / 'data.jsonl'
    data.write_text('{"fixture": 1}\n{"fixture": 2}\n')
    monkeypatch.setitem(status.DATASET_CONFIGS, 'trans_review',
                        {'data_files': dict(train=str(data), validation=str(data), test=str(data))})
    argv = [sys.executable, '-m', 'src.train', '--model', 'codet5', '--method', 'lora',
            '--experiment_id', '0', '--subset', '0.5', '--dataset', 'trans_review',
            '--results-root', str(tmp_path), '--memory-profile', 'a100', '--gen-batch-size', '32']
    args = build_parser().parse_args(argv[3:])
    model, method, spec = get_spec(args.model, args.method)
    spec, overrides = apply_memory_profile(spec, resolve_protocol_overrides(model, args),
                                          'a100', model, method)
    run = resolve_run(spec, experiment_id=0, results_root=args.results_root)
    root = Path(run['experiment_results_dir']); root.mkdir()
    identity = recovery_identity(spec, overrides, args, run)
    stamp = fingerprint(identity)
    (root / 'run_identity.json').write_text(json.dumps(dict(identity=identity, fingerprint=stamp)))
    result = {k: 0 for k in ['test_bleu_4', 'test_exact_match', 'test_rouge', 'test_bertscore', 'test_meteor']}
    result.update(run_fingerprint=stamp, expected_test_rows=1,
                  raw_data=[dict(prediction='x', reference='y')])
    path = root / 'complete_results.json'
    path.write_text(json.dumps(result))
    assert status.training_result(argv, 'D') == path
    result.update(expected_test_rows=2, raw_data=result['raw_data'] * 2)
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match='coverage incomplete'):
        status.training_result(argv, 'D')
