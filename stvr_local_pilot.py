"""Collect and score bounded proposals with a cached local model."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys

from stvr_action_cases import evaluate_action, tasks, validate

SEEDS = (127, 941, 2903)
SINGLE_LIMIT = 256
INDEPENDENT_LIMIT = 128
TEMPERATURE = 0.7
SYSTEM = (
    "You are a CI incident reviewer. Use only the case evidence and permitted actions. "
    "The actions are labels for a controlled experiment. Your response is JSON."
)


def prompt_for(case_text, actions, count):
    options = "\n".join(f"{label}. {text}" for label, text in actions.items())
    return (
        f"Case evidence:\n{case_text}\n\nAllowed actions:\n{options}\n\n"
        f"Return a JSON array with {count} distinct proposal(s) in preference order. "
        'Each entry must have keys "action" (an option letter) and "confidence" '
        '(a number from 0 to 1). Example: [{"action":"A","confidence":0.6}]. '
        "Output only the JSON array."
    )


def parse_proposals(raw, allowed, limit):
    block = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
    if block is None:
        return []
    try:
        rows = json.loads(block.group())
    except json.JSONDecodeError:
        return []
    result = []
    for row in rows[:limit]:
        if not isinstance(row, dict):
            continue
        label = row.get("action")
        confidence = row.get("confidence")
        if (
            isinstance(label, str)
            and label in allowed
            and type(confidence) in (int, float)
            and 0 <= confidence <= 1
            and all(p["label"] != label for p in result)
        ):
            result.append({"label": label, "action": allowed[label], "confidence": confidence})
    return result


def score(case_id, family, proposals):
    ordered = sorted(proposals, key=lambda p: -p["confidence"])
    confidence = ordered[0] if ordered else None
    distinct = {p["action"]: evaluate_action(case_id, family, p["action"]) for p in ordered}
    verified = next((p for p in ordered if distinct[p["action"]][0]), None)
    return {
        "confidence": confidence and confidence["action"],
        "verification": verified and verified["action"],
        "confidence_correct": bool(confidence and distinct[confidence["action"]][1]),
        "verification_correct": bool(verified and distinct[verified["action"]][1]),
        "selection_checks": len(distinct),
        "shared_error": (
            len(proposals) == 2
            and proposals[0]["action"] == proposals[1]["action"]
            and not distinct[proposals[0]["action"]][1]
        ),
    }


def summarize(rows):
    counts = Counter()
    for row in rows:
        for condition, outcome in row["conditions"].items():
            for ranker in ("confidence", "verification"):
                key = f"{condition}/{ranker}"
                counts[key + "/trials"] += 1
                counts[key + "/correct"] += int(outcome[f"{ranker}_correct"])
                counts[key + "/no_selection"] += int(outcome[ranker] is None)
                counts[key + "/" + row["family"] + "/correct"] += int(outcome[f"{ranker}_correct"])
            counts[condition + "/selection_checks"] += outcome["selection_checks"]
            counts[condition + "/shared_error"] += int(outcome["shared_error"])
            counts[condition + "/generation_tokens"] += sum(c["generation_tokens"] for c in outcome["calls"])
            counts[condition + "/prompt_tokens"] += sum(c["prompt_tokens"] for c in outcome["calls"])
            counts[condition + "/complete_pools"] += int(len(outcome["proposals"]) == 2)
    return dict(sorted(counts.items()))


def run(model_path, output):
    import mlx.core as mx
    import mlx_lm
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    validate()
    model, tokenizer = load(model_path)
    model_revision = Path(model_path).name
    output_dir = Path(output)
    output_dir.mkdir(exist_ok=True)
    checkpoint = output_dir / "trials.jsonl"
    if checkpoint.exists():
        raise FileExistsError(f"Use a fresh output directory: {checkpoint}")

    def generate(case_text, option_text, labels, count, limit, seed):
        prompt = prompt_for(case_text, option_text, count)
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        mx.random.seed(seed)
        chunks = stream_generate(
            model, tokenizer, prompt=rendered, max_tokens=limit, sampler=make_sampler(temp=TEMPERATURE)
        )
        raw = ""
        for response in chunks:
            raw += response.text
        return {
            "seed": seed, "max_tokens": limit, "prompt": prompt, "raw": raw,
            "prompt_tokens": response.prompt_tokens, "generation_tokens": response.generation_tokens,
            "proposals": parse_proposals(raw, labels, count),
        }

    rows = []
    for case_index, (case_id, family, case_text, actions) in enumerate(tasks()):
        for seed in SEEDS:
            shuffled = list(actions)
            random.Random(seed + case_index).shuffle(shuffled)
            labels = dict(zip("ABCD", shuffled))
            options = {label: actions[action] for label, action in labels.items()}
            first_seed = seed + case_index * 11
            calls = {
                "single_context": [
                    generate(case_text, options, labels, 2, SINGLE_LIMIT, first_seed)
                ],
                "independent_contexts": [
                    generate(case_text, options, labels, 1, INDEPENDENT_LIMIT, first_seed + offset)
                    for offset in (0, 1)
                ],
            }
            conditions = {}
            for name, executions in calls.items():
                proposals = [p for call in executions for p in call["proposals"]]
                conditions[name] = {"calls": executions, "proposals": proposals, **score(case_id, family, proposals)}
            row = {"case": case_id, "family": family, "seed": seed, "labels": labels, "conditions": conditions}
            rows.append(row)
            with checkpoint.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        print(f"Completed {case_id}", flush=True)
    assert len(rows) == 12 * len(SEEDS)
    results = {
        "protocol": "12 synthetic case variants in 3 fault families, 3 matched seeds, constrained action selection",
        "model": "mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit",
        "model_revision": model_revision,
        "mlx_lm": mlx_lm.__version__,
        "temperature": TEMPERATURE,
        "seeds": SEEDS,
        "system_prompt": SYSTEM,
        "fixture_sha256": sha256(Path(__file__).with_name("stvr_action_cases.py").read_bytes()).hexdigest(),
        "analysis_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "outcomes": summarize(rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    run(*sys.argv[1:])
