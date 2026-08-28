"""ICL result filename builder (phase 04).

Legacy files keep their exact name when every new flag is at its default;
any non-default run gets an explicit demo-seed / prompt-version suffix so
revision outputs can never overwrite legacy artifacts.
"""


def icl_result_filename(model_name, num_examples, selection_method, *,
                        demo_seed=None, prompt_version="v1", non_default=False):
    base = f"{model_name}_icl_{num_examples}ex_{selection_method}"
    if demo_seed is None and prompt_version == "v1" and not non_default:
        return base + ".json"
    parts = [base]
    if demo_seed is not None:
        parts.append(f"s{demo_seed}")
    parts.append(prompt_version)
    return "_".join(parts) + ".json"
