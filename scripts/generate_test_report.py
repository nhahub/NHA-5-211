"""Run Ruff and pytest and regenerate reports/test_report.md from real command outputs."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.model_registry import compute_file_sha256

_LFS_POINTER_SUFFIXES = {".keras", ".pkl", ".h5", ".onnx", ".npz", ".parquet"}
_LFS_POINTER_EXACT = {"models/xgb_model.json"}

_TEST_CASE_RE = re.compile(
    r"^(tests/[^:\s]+)::([^\s]+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAILED|XPASSED)(?=\s|$|\(|[A-Za-z_])",
    re.MULTILINE,
)
_DURATION_RE = re.compile(
    r"^\s*([0-9]+\.[0-9]+)s\s+(call|setup|teardown)\s+(tests/[^:\s]+)::([^\s]+)\s*$",
    re.MULTILINE,
)
_PYTEST_SUMMARY_RE = re.compile(
    r"=+\s+([0-9]+[^\n=]*?)\s+in\s+([0-9]+\.[0-9]+)s(?:\s+\([^)]*\))?\s*=+\s*$",
    re.MULTILINE,
)


@dataclass
class CommandRun:
    """Captured command invocation and measured output."""

    command: list[str]
    display_command: str
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float

    @property
    def combined_output(self) -> str:
        if self.stdout and self.stderr:
            return f"{self.stdout.rstrip()}\n{self.stderr}"
        return self.stdout or self.stderr


@dataclass
class ParsedTestCase:
    """Single parsed test outcome and timing from pytest verbose + durations output."""

    module: str
    name: str
    status: str
    call_sec: float
    total_sec: float


def _run_tool(
    cmd: list[str],
    display_command: str,
    cwd: Path = PROJECT_ROOT,
    env: dict[str, str] | None = None,
) -> CommandRun:
    t0 = time.perf_counter()
    proc_env = {**(env or os.environ), "COLUMNS": "240"}
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=proc_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - t0
    return CommandRun(
        command=cmd,
        display_command=display_command,
        returncode=int(proc.returncode),
        stdout=proc.stdout or "",
        stderr=proc.stderr or "",
        duration_sec=float(elapsed),
    )


def prepare_stub_ci_copy(src_root: Path, dest_dir: Path) -> list[str]:
    """Copy git-tracked files to ``dest_dir`` and replace LFS model binaries with pointer stubs."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=str(src_root),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode == 0 and proc.stdout.strip():
        rel_files = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    else:
        ignored = {".git", ".venv", ".pytest_cache", ".ruff_cache", "__pycache__"}
        rel_files = [
            p.relative_to(src_root).as_posix()
            for p in src_root.rglob("*")
            if p.is_file() and not any(part in ignored for part in p.parts)
        ]

    replaced_stubs: list[str] = []
    for rel in rel_files:
        src = src_root / rel
        if not src.is_file():
            continue
        dst = dest_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        rel_norm = rel.replace("\\", "/")
        is_lfs_target = rel_norm.startswith("models/") and (
            src.suffix.lower() in _LFS_POINTER_SUFFIXES or rel_norm in _LFS_POINTER_EXACT
        )
        if is_lfs_target:
            sha = compute_file_sha256(src)
            size = int(src.stat().st_size)
            dst.write_text(
                f"version https://git-lfs.github.com/spec/v1\noid sha256:{sha}\nsize {size}\n",
                encoding="utf-8",
            )
            replaced_stubs.append(rel_norm)
        else:
            shutil.copy2(src, dst)
    return sorted(replaced_stubs)


def parse_pytest_output(output: str) -> tuple[list[ParsedTestCase], str, float | None, dict[str, int]]:
    """Parse pytest -v --durations=0 output into structured per-test results and totals."""
    call_durations: dict[tuple[str, str], float] = {}
    total_durations: dict[tuple[str, str], float] = {}

    for sec_str, phase, module, name in _DURATION_RE.findall(output):
        key = (module.replace("\\", "/"), name)
        sec = float(sec_str)
        total_durations[key] = total_durations.get(key, 0.0) + sec
        if phase == "call":
            call_durations[key] = sec

    cases: list[ParsedTestCase] = []
    counts = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "other": 0}
    for module_raw, name, status in _TEST_CASE_RE.findall(output):
        module = module_raw.replace("\\", "/")
        key = (module, name)
        cases.append(
            ParsedTestCase(
                module=module,
                name=name,
                status=status,
                call_sec=round(call_durations.get(key, 0.0), 2),
                total_sec=round(total_durations.get(key, 0.0), 2),
            )
        )
        lower_st = status.lower()
        if lower_st in counts:
            counts[lower_st] += 1
        else:
            counts["other"] += 1

    summary_match = None
    for match in _PYTEST_SUMMARY_RE.finditer(output):
        summary_match = match
    if summary_match:
        summary_text = summary_match.group(1).strip()
        summary_duration: float | None = float(summary_match.group(2))
    else:
        summary_text = f"{counts['passed']} passed, {counts['failed']} failed, {counts['skipped']} skipped"
        summary_duration = None

    return cases, summary_text, summary_duration, counts


def render_test_report(
    *,
    ruff_version: str,
    pytest_version: str,
    ruff_check: CommandRun,
    ruff_format: CommandRun,
    pytest_run: CommandRun,
    stub_pytest_run: CommandRun | None = None,
    stub_files_count: int = 0,
    generated_utc: str | None = None,
) -> str:
    """Render reports/test_report.md cleanly from observed full-artifact and LFS-stub runs."""
    timestamp = generated_utc or datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    check_lines = [ln.strip() for ln in ruff_check.combined_output.splitlines() if ln.strip()]
    format_lines = [ln.strip() for ln in ruff_format.combined_output.splitlines() if ln.strip()]

    check_result = check_lines[-1] if check_lines else ("PASSED" if ruff_check.returncode == 0 else "FAILED")
    format_result = (
        format_lines[-1] if format_lines else ("PASSED" if ruff_format.returncode == 0 else "FAILED")
    )

    cases, summary_text, reported_duration, counts = parse_pytest_output(pytest_run.combined_output)
    duration_display = (
        f"{reported_duration:.2f}s" if reported_duration is not None else f"{pytest_run.duration_sec:.2f}s"
    )

    lines = [
        "# SMART IDS v3 — Automated Verification & Test Report",
        "",
        f"> Generated automatically by `scripts/generate_test_report.py` on `{timestamp}`.",
        "",
        f"## 1. Static Analysis & Formatting (`{ruff_version}`)",
        f"- **Command:** `{ruff_check.display_command}`",
        f"  - **Status:** `{'PASSED' if ruff_check.returncode == 0 else 'FAILED'}` (`exit={ruff_check.returncode}`, `{ruff_check.duration_sec:.2f}s`) — `{check_result}`",
        f"- **Command:** `{ruff_format.display_command}`",
        f"  - **Status:** `{'PASSED' if ruff_format.returncode == 0 else 'FAILED'}` (`exit={ruff_format.returncode}`, `{ruff_format.duration_sec:.2f}s`) — `{format_result}`",
        "",
        "---",
        "",
        "## 2. CI Verification Matrix (Real Artifacts vs. LFS Pointer Stub Simulation)",
        "",
        "| Suite Mode | Collected | Passed | Failed | Error | Skipped | Duration_s | Status |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
        (
            f"| Full Artifacts (local binaries) | {len(cases)} | {counts['passed']} | {counts['failed']} | "
            f"{counts['error']} | {counts['skipped']} | {duration_display.rstrip('s')} | "
            f"{'PASSED' if pytest_run.returncode == 0 else 'FAILED'} |"
        ),
    ]

    stub_cases: list[ParsedTestCase] = []
    stub_summary_text = ""
    stub_duration_display = ""
    stub_counts: dict[str, int] = {"passed": 0, "failed": 0, "error": 0, "skipped": 0, "other": 0}
    if stub_pytest_run is not None:
        stub_cases, stub_summary_text, stub_reported_dur, stub_counts = parse_pytest_output(
            stub_pytest_run.combined_output
        )
        stub_duration_display = (
            f"{stub_reported_dur:.2f}s"
            if stub_reported_dur is not None
            else f"{stub_pytest_run.duration_sec:.2f}s"
        )
        lines.append(
            f"| LFS Pointer Stubs (`lfs: false` mirror, {stub_files_count} stubs) | {len(stub_cases)} | "
            f"{stub_counts['passed']} | {stub_counts['failed']} | {stub_counts['error']} | "
            f"{stub_counts['skipped']} | {stub_duration_display.rstrip('s')} | "
            f"{'PASSED' if stub_pytest_run.returncode == 0 else 'FAILED'} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 3. Full-Artifact Test Suite (`{pytest_version}`)",
            f"- **Command:** `{pytest_run.display_command}`",
            (
                f"- **Summary:** **`{len(cases)}` collected — `{summary_text}` (`{duration_display}` total, "
                f"{counts['passed']} passed / {counts['failed']} failed / {counts['error']} error / {counts['skipped']} skipped)**"
            ),
            "",
            "| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |",
            "|---:|---|---|---|---:|---:|",
        ]
    )
    for idx, tc in enumerate(cases, start=1):
        lines.append(
            f"| {idx} | `{tc.module}` | `{tc.name}` | {tc.status} | {tc.call_sec:.2f} | {tc.total_sec:.2f} |"
        )

    if stub_pytest_run is not None:
        skipped_names = [f"`{tc.module}::{tc.name}`" for tc in stub_cases if tc.status == "SKIPPED"]
        lines.extend(
            [
                "",
                "---",
                "",
                f"## 4. LFS Pointer Stub Simulation Suite (`{pytest_version}`, `lfs: false` Mirror)",
                f"- **Command:** `{stub_pytest_run.display_command}` (`{stub_files_count}` LFS binaries replaced by pointer stubs)",
                (
                    f"- **Summary:** **`{len(stub_cases)}` collected — `{stub_summary_text}` (`{stub_duration_display}` total, "
                    f"{stub_counts['passed']} passed / {stub_counts['failed']} failed / {stub_counts['error']} error / {stub_counts['skipped']} skipped)**"
                ),
                (
                    f"- **Skipped Tests (stub detection):** {', '.join(skipped_names)}"
                    if skipped_names
                    else "- **Skipped Tests (stub detection):** None"
                ),
                "",
                "| # | Test Module | Test Case | Status | Call_Duration_s | Total_Duration_s |",
                "|---:|---|---|---|---:|---:|",
            ]
        )
        for idx, tc in enumerate(stub_cases, start=1):
            lines.append(
                f"| {idx} | `{tc.module}` | `{tc.name}` | {tc.status} | {tc.call_sec:.2f} | {tc.total_sec:.2f} |"
            )

    lines.append("")
    return "\n".join(lines)


def generate_test_report(
    output_path: Path = PROJECT_ROOT / "reports" / "test_report.md",
    python_exe: str = sys.executable,
    include_stub_sim: bool = True,
) -> tuple[Path, int]:
    """Execute Ruff and pytest suite (full artifacts + optional LFS stub simulation) and persist report."""
    ruff_ver_run = _run_tool([python_exe, "-m", "ruff", "--version"], "ruff --version")
    pytest_ver_run = _run_tool([python_exe, "-m", "pytest", "--version"], "pytest --version")

    ruff_version = ruff_ver_run.stdout.strip().splitlines()[0] if ruff_ver_run.stdout.strip() else "ruff"
    pytest_version = (
        pytest_ver_run.stdout.strip().splitlines()[0] if pytest_ver_run.stdout.strip() else "pytest"
    )

    ruff_check = _run_tool(
        [python_exe, "-m", "ruff", "check", "src", "dashboard", "scripts", "tests", "app.py"],
        "ruff check src dashboard scripts tests app.py",
    )
    ruff_format = _run_tool(
        [
            python_exe,
            "-m",
            "ruff",
            "format",
            "--check",
            "src",
            "dashboard",
            "scripts",
            "tests",
            "app.py",
        ],
        "ruff format --check src dashboard scripts tests app.py",
    )
    pytest_run = _run_tool(
        [python_exe, "-m", "pytest", "tests/", "-v", "--durations=0", "--tb=short"],
        "pytest tests/ -v --durations=0 --tb=short",
    )

    cases, _, _, _ = parse_pytest_output(pytest_run.combined_output)
    if not cases:
        raise RuntimeError(
            "Unable to parse any pytest test cases from output; refusing to write incomplete test_report.md."
        )

    stub_pytest_run: CommandRun | None = None
    stub_files_count = 0
    if include_stub_sim:
        with tempfile.TemporaryDirectory(prefix="smart_ids_stub_ci_") as tmp_dir:
            sim_root = Path(tmp_dir)
            replaced = prepare_stub_ci_copy(PROJECT_ROOT, sim_root)
            stub_files_count = len(replaced)
            sim_env = {**os.environ, "PYTHONPATH": str(sim_root)}
            stub_pytest_run = _run_tool(
                [python_exe, "-m", "pytest", "tests/", "-v", "--durations=0", "--tb=short"],
                "pytest tests/ -v --durations=0 --tb=short (stub-mode simulation)",
                cwd=sim_root,
                env=sim_env,
            )

    markdown = render_test_report(
        ruff_version=ruff_version,
        pytest_version=pytest_version,
        ruff_check=ruff_check,
        ruff_format=ruff_format,
        pytest_run=pytest_run,
        stub_pytest_run=stub_pytest_run,
        stub_files_count=stub_files_count,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    stub_ok = stub_pytest_run is None or stub_pytest_run.returncode == 0
    overall_rc = (
        0
        if (
            ruff_check.returncode == 0
            and ruff_format.returncode == 0
            and pytest_run.returncode == 0
            and stub_ok
        )
        else 1
    )
    return output_path, overall_rc


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate reports/test_report.md from Ruff and pytest.")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "test_report.md",
        help="Target Markdown report path.",
    )
    parser.add_argument(
        "--no-stub-sim",
        action="store_true",
        help="Skip secondary LFS pointer stub CI simulation run.",
    )
    args = parser.parse_args()
    out_path, rc = generate_test_report(output_path=args.output, include_stub_sim=not args.no_stub_sim)
    print(f"Wrote {out_path} (overall exit={rc})")
    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
