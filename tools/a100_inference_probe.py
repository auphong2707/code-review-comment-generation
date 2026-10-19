#!/usr/bin/env python3
"""Bounded ICL/judge VRAM pilots on the allocated A100. Execute only on the server GPU."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.a100_preflight import runtime_digest, environment_identity, check_report, require_gpu, hardware_policy
from src.training.recovery import atomic_json, install_stop_handlers, stop_if_requested

JUDGES = {
    "qwen-judge": "assets/models/llm-judge/Qwen2.5-Coder-3B-Instruct",
    "llama-judge": "assets/models/llm-judge/Llama-3.1-8B-Instruct",
}
ICL_MODELS = ("qwen25-0.5b", "qwen25", "qwen15")


def run_probe(args, digest):
    hardware = require_gpu()
    import torch
    torch.cuda.reset_peak_memory_stats()
    install_stop_handlers()
    if args.kind == "icl":
        from src.icl_inference import load_base_model_and_tokenizer
        model, tokenizer = load_base_model_and_tokenizer(args.model, "sdpa")
        output_tokens, beams = 256, 3
    else:
        from src.llm_as_judge.config import JudgeConfig
        from src.llm_as_judge.qwen_judge import make_judge
        judge = make_judge(JudgeConfig(model_path=JUDGES[args.model], k=3, batch_size=1,
                          backend="qwen" if args.model == "qwen-judge" else "hf-chat",
                          local_files_only=True, input_token_limit=args.max_input_tokens))
        judge._load()
        model, tokenizer = judge.model, judge.tokenizer
        output_tokens, beams = 32, 1
    if any(device in ("cpu", "disk") for device in getattr(model, "hf_device_map", {}).values()):
        raise RuntimeError("Probe requires a GPU-resident model, not CPU/disk offload")
    model.eval()
    token = tokenizer.encode("code", add_special_tokens=False)[0]
    input_ids = torch.full((args.batch_size, args.max_input_tokens), token, dtype=torch.long, device=model.device)
    mask = torch.ones_like(input_ids)
    kwargs = dict(max_new_tokens=output_tokens, min_new_tokens=output_tokens,
                  num_beams=beams, do_sample=args.kind == "judge",
                  pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id)
    if args.kind == "judge":
        kwargs.update(temperature=0.3, top_p=0.9, top_k=20, repetition_penalty=1.05)
    # Exercise allocation and warm reuse at the exact maximum admitted input/output lengths.
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if args.kind == "icl" else nullcontext()
    with torch.inference_mode(), autocast:
        for _ in range(3):
            stop_if_requested()
            generated = model.generate(input_ids=input_ids, attention_mask=mask, **kwargs)
            torch.cuda.synchronize()
            del generated
    stop_if_requested()
    peak = torch.cuda.max_memory_reserved() / 2**30
    return {"runtime_digest": digest, "environment": environment_identity(), "hardware": hardware,
            "kind": args.kind, "model": args.model, "max_input_tokens": args.max_input_tokens,
            "max_output_tokens": output_tokens, "batch_size": args.batch_size, "num_beams": beams,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "peak_reserved_gib": peak, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "total_memory_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "admitted": peak <= hardware_policy()[3],
            "note": "Maximum-length synthetic memory pilot; no quality scores or campaign claims"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["icl", "judge"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--max-input-tokens", type=int)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="ICL generation batch the campaign uses (judges stay at 1)")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    if args.kind == "icl" and args.model not in ICL_MODELS or args.kind == "judge" and args.model not in JUDGES:
        parser.error("Unsupported model for this probe kind")
    args.max_input_tokens = ((2048 if args.kind == "icl" else 4096)
                             if args.max_input_tokens is None else args.max_input_tokens)
    if args.max_input_tokens < 1 or args.kind == "icl" and args.max_input_tokens != 2048:
        parser.error("ICL qualification uses exactly 2048 input tokens; judge budget must be positive")
    if args.batch_size < 1 or args.kind == "judge" and args.batch_size != 1:
        parser.error("Batch size must be positive; judges run at batch 1")
    digest = runtime_digest()
    path = Path(args.out_root) / "admissions" / f"{args.kind}_{args.model}_{args.max_input_tokens}.json"
    expected = {"model": args.model, "kind": args.kind, "max_input_tokens": args.max_input_tokens,
                "batch_size": args.batch_size}
    if args.check:
        check_report(json.loads(path.read_text()), digest, expected)
        print(f"Qualified: {args.kind}/{args.model}/{args.max_input_tokens} tokens")
        return
    report = run_probe(args, digest)
    atomic_json(path, report)
    check_report(report, digest, expected)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
