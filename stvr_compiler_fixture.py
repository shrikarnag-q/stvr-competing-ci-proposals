"""Validate an isolated compiler task and its independent verification checks."""

from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

MAIN = """#include <stdio.h>
int rate(void);
int main(void) {
    printf("%d\\n", rate());
    return 0;
}
"""
RATE = """int rate(void) {
#ifdef ALT_TARGET
    return 11;
#else
    return 7;
#endif
}
"""
CI = """from pathlib import Path
import subprocess
import sys

cc, target = sys.argv[1:]
flags = ["-DALT_TARGET"] if target == "alternate" else []
binary = Path("out")
binary.unlink(missing_ok=True)
subprocess.run([cc, *flags, "main.c", "-o", str(binary)], check=True)
subprocess.run(["./out"], check=True)
"""
FIXTURE = {"main.c": MAIN, "rate.c": RATE, "ci.py": CI}
# Kept in the independent evaluator, outside each proposal's workspace.
EXPECTED_OUTPUT = {"default": "7", "alternate": "11"}


def run_check(workspace, compiler, target):
    result = subprocess.run(
        [sys.executable, "ci.py", compiler, target],
        cwd=workspace, capture_output=True, text=True, timeout=20, check=False,
    )
    binary = (workspace / "out").exists()
    stdout = result.stdout.strip()
    return {
        "exit_code": result.returncode,
        "linked_binary": binary,
        "unresolved_rate": (
            "rate" in result.stderr and
            any(s in result.stderr.lower() for s in ("undefined", "unresolved"))
        ),
        "output": stdout,
        "passed": result.returncode == 0 and binary and stdout == EXPECTED_OUTPUT[target],
    }


def evaluate(compiler, edits):
    with TemporaryDirectory(prefix="stvr-compiler-") as temp:
        workspace = Path(temp)
        for name, source in (FIXTURE | edits).items():
            (workspace / name).write_text(source, encoding="utf-8")
        visible = run_check(workspace, compiler, "default")
        withheld = run_check(workspace, compiler, "alternate")
    return {"visible": visible, "withheld": withheld, "verified": visible["passed"] and withheld["passed"]}


def main():
    compiler = shutil.which("cc")
    if compiler is None:
        raise SystemExit("A C compiler named cc is required")
    baseline = evaluate(compiler, {})
    retry = evaluate(compiler, {})
    narrow = evaluate(compiler, {"main.c": MAIN.replace("int rate(void);", "int rate(void) { return 7; }")})
    repair = evaluate(compiler, {"ci.py": CI.replace('"main.c", "-o"', '"main.c", "rate.c", "-o"')})

    assert baseline == retry
    assert all(
        baseline[k]["unresolved_rate"] and not baseline[k]["linked_binary"]
        for k in ("visible", "withheld")
    )
    assert narrow["visible"]["passed"] and not narrow["withheld"]["passed"]
    assert narrow["withheld"]["exit_code"] == 0 and narrow["withheld"]["output"] == "7"
    assert repair["verified"] and repair["withheld"]["output"] == "11"

    version = subprocess.check_output([compiler, "--version"], text=True, timeout=10)
    print(json.dumps({
        "task": "missing source in compiler command",
        "origin": "locally generated synthetic compiler fixture",
        "execution": "authored proposals checked in disposable workspaces",
        "compiler": version.splitlines()[0],
        "python": sys.version.split()[0],
        "fixture_sha256": {name: sha256(source.encode()).hexdigest() for name, source in FIXTURE.items()},
        "expected_output": EXPECTED_OUTPUT,
        "results": {"unmodified": baseline, "retry": retry, "narrow_repair": narrow, "restore_source": repair},
    }, indent=2))


if __name__ == "__main__":
    main()
