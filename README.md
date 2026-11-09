# Reproducibility materials for When AI Employees Compete

The pilot uses 12 synthetic CI case variants across three fault families and three matched sampling seeds per variant. The action menus and scoring logic were specified before this run. Models choose action labels. The evaluator runs only authored fixtures. The result is a local single-model pilot, not an operational Tymeline evaluation.

Requirements for rerunning generation include an Apple Silicon Mac, a C compiler available as cc, Python 3.12, MLX LM 0.32.2, and the following public model snapshot at revision c4d79ba1901d99806ef757642a552acebb851a35.
https://huggingface.co/mlx-community/NVIDIA-Nemotron-3-Nano-4B-4bit/tree/c4d79ba1901d99806ef757642a552acebb851a35

From the extracted directory, use a Python environment with MLX LM installed:

    python stvr_compiler_fixture.py
    python stvr_local_pilot.py /absolute/path/to/model/snapshot /absolute/path/to/fresh-output

Recorded trial rows contain the exact user prompts, option maps, sampling seeds, raw responses, token counts and decisions. The system prompt and model configuration are in summary.json. Model outputs can vary with the runtime or hardware. To verify reported counts from the archived rows without running the model, use:

    python -c 'import json; from stvr_local_pilot import summarize; rows=[json.loads(s) for s in open("trials.jsonl")]; assert summarize(rows)==json.load(open("summary.json"))["outcomes"]'

The compiler fault variants share one executable fixture. The numerical regression variants share a clamp contract. Missing-evidence cases evaluate predefined escalation requests. The unit of this pilot is a case-seed pair. Repeated seeds do not represent independent incidents.
