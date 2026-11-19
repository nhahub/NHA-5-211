"""Verify that numeric metrics and test claims in README.md match generated reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_MODEL_KEYS: dict[str, tuple[str, str, str]] = {
    "XGBoost": ("XGBoost", "XGBoost", "XGBoost"),
    "1D-CNN": ("CNN", "1D-CNN", "1D-CNN"),
    "MLP": ("MLP", "MLP", "MLP"),
}


def _check_close(
    errors: list[str],
    label: str,
    found_val: float,
    expected_val: Any,
    tol: float = 0.00015,
) -> None:
    if expected_val is None:
        errors.append(f"{label}: expected value missing in evaluation_summary.json")
        return
    exp = float(expected_val)
    if abs(found_val - exp) > tol:
        errors.append(f"{label}: README/doc has {found_val:.4f}, generated report has {exp:.4f}")


def parse_test_report_counts(test_report_text: str) -> dict[str, int] | None:
    """Extract passed/failed/skipped counts from reports/test_report.md."""
    match = re.search(
        r"(\d+)\s+passed,\s+(\d+)\s+failed,\s+(\d+)\s+skipped",
        test_report_text,
    )
    if not match:
        return None
    return {
        "passed": int(match.group(1)),
        "failed": int(match.group(2)),
        "skipped": int(match.group(3)),
    }


def check_readme_numbers(
    readme_path: Path | None = None,
    summary_path: Path | None = None,
    test_report_path: Path | None = None,
    results_details_path: Path | None = None,
    max_readme_lines: int = 150,
) -> list[str]:
    """Return a list of mismatch descriptions (empty list when everything matches)."""
    rm_path = readme_path or (PROJECT_ROOT / "README.md")
    sum_path = summary_path or (PROJECT_ROOT / "reports" / "evaluation_summary.json")
    tr_path = test_report_path or (PROJECT_ROOT / "reports" / "test_report.md")
    rd_path = results_details_path or (PROJECT_ROOT / "docs" / "results_details.md")

    errors: list[str] = []
    if not rm_path.exists():
        return [f"Missing README file: {rm_path}"]
    if not sum_path.exists():
        return [f"Missing generated summary: {sum_path}"]

    readme_text = rm_path.read_text(encoding="utf-8")
    readme_lines = readme_text.splitlines()
    if len(readme_lines) > max_readme_lines:
        errors.append(f"README.md line count {len(readme_lines)} exceeds limit {max_readme_lines}")

    summary = json.loads(sum_path.read_text(encoding="utf-8"))
    sup = summary.get("supervised_evaluations", {})
    tb_models = summary.get("time_block_protocol", {}).get("models", {})
    bd_models = summary.get("by_day_protocol", {}).get("multiclass_models", {})

    # 1. Verify Results Overview table F1_Macro + 95% CIs for XGBoost, 1D-CNN, MLP
    row_re = re.compile(
        r"^\|\s*\*\*(?P<name>XGBoost|1D-CNN|MLP)\b[^|]*\|\s*`(?P<imb>[^`]+)`\s*\|\s*"
        r"(?:\*\*)?(?P<rnd>[0-9.]+)(?:\*\*)?\s*`\[(?P<rnd_lo>[0-9.]+),\s*(?P<rnd_hi>[0-9.]+)\]`\s*\|\s*"
        r"(?:\*\*)?(?P<tb>[0-9.]+)(?:\*\*)?\s*`\[(?P<tb_lo>[0-9.]+),\s*(?P<tb_hi>[0-9.]+)\]`\s*\|\s*"
        r"(?:\*\*)?(?P<bd>[0-9.]+)(?:\*\*)?\s*\|"
    )
    matched_models: set[str] = set()
    for line in readme_lines:
        m = row_re.match(line.strip())
        if not m:
            continue
        name = m.group("name")
        matched_models.add(name)
        sup_k, tb_k, bd_k = _MODEL_KEYS[name]

        _check_close(
            errors, f"{name} random F1_Macro", float(m.group("rnd")), sup.get(sup_k, {}).get("f1_macro")
        )
        rnd_ci = sup.get(sup_k, {}).get("f1_macro_ci95", {})
        _check_close(errors, f"{name} random CI low", float(m.group("rnd_lo")), rnd_ci.get("ci95_low"))
        _check_close(errors, f"{name} random CI high", float(m.group("rnd_hi")), rnd_ci.get("ci95_high"))

        _check_close(
            errors,
            f"{name} time_block F1_Macro",
            float(m.group("tb")),
            tb_models.get(tb_k, {}).get("f1_macro"),
        )
        tb_ci = tb_models.get(tb_k, {}).get("f1_macro_ci95", {})
        _check_close(errors, f"{name} time_block CI low", float(m.group("tb_lo")), tb_ci.get("ci95_low"))
        _check_close(errors, f"{name} time_block CI high", float(m.group("tb_hi")), tb_ci.get("ci95_high"))

        _check_close(
            errors,
            f"{name} by_day closed F1_Macro",
            float(m.group("bd")),
            bd_models.get(bd_k, {}).get("f1_macro"),
        )

    for expected_name in _MODEL_KEYS:
        if expected_name not in matched_models:
            errors.append(f"README.md Results Overview table is missing row for '{expected_name}'")

    # 2. Verify F1_Macro_Reliable callout (0.9972 / 0.9562 / 0.9286)
    rel_match = re.search(
        r"F1_Macro_Reliable.*?`([0-9.]+)`\s*/\s*`([0-9.]+)`\s*/\s*`([0-9.]+)`",
        readme_text,
    )
    if rel_match:
        _check_close(
            errors,
            "XGBoost F1_Macro_Reliable",
            float(rel_match.group(1)),
            sup.get("XGBoost", {}).get("f1_macro_reliable"),
        )
        _check_close(
            errors,
            "1D-CNN F1_Macro_Reliable",
            float(rel_match.group(2)),
            sup.get("CNN", {}).get("f1_macro_reliable"),
        )
        _check_close(
            errors,
            "MLP F1_Macro_Reliable",
            float(rel_match.group(3)),
            sup.get("MLP", {}).get("f1_macro_reliable"),
        )

    # 3. Verify raw & clean flow counts if present in Results Overview header
    count_match = re.search(r"`([0-9,]+)`\s*Raw\s*→\s*`([0-9,]+)`\s*Clean", readme_text)
    if count_match:
        raw_val = int(count_match.group(1).replace(",", ""))
        clean_val = int(count_match.group(2).replace(",", ""))
        exp_raw = summary.get("dataset_source", {}).get("raw_row_count")
        exp_clean = (
            summary.get("preprocessing_audit", {}).get("cleaning_details", {}).get("cleaned_row_count")
        )
        if exp_raw is not None and raw_val != int(exp_raw):
            errors.append(f"README raw row count {raw_val} != {exp_raw}")
        if exp_clean is not None and clean_val != int(exp_clean):
            errors.append(f"README clean row count {clean_val} != {exp_clean}")

    # 4. Verify Autoencoder & key summary figures in README Key Findings
    ae_match = re.search(r"Autoencoder.*?F1\s*=\s*([0-9.]+).*?ROC-AUC\s*=\s*([0-9.]+)", readme_text)
    if ae_match:
        ae_ev = summary.get("autoencoder_evaluation", {})
        _check_close(errors, "README Autoencoder F1", float(ae_match.group(1)), ae_ev.get("f1"))
        _check_close(errors, "README Autoencoder ROC-AUC", float(ae_match.group(2)), ae_ev.get("roc_auc"))

    by_day_bin = (
        summary.get("by_day_protocol", {})
        .get("binary_cross_day_models", {})
        .get("XGBoost (1 - P_Benign)", {})
    )
    bd_rec_match = re.search(r"detects only `([0-9.]+)%` of unseen attacks \(`F1 = ([0-9.]+)`\)", readme_text)
    if bd_rec_match and by_day_bin:
        _check_close(
            errors,
            "README by_day unseen recall %",
            float(bd_rec_match.group(1)) / 100.0,
            by_day_bin.get("recall"),
            tol=0.0005,
        )
        _check_close(errors, "README by_day binary F1", float(bd_rec_match.group(2)), by_day_bin.get("f1"))

    # 5. Verify test counts against reports/test_report.md
    if tr_path.exists():
        tr_text = tr_path.read_text(encoding="utf-8")
        counts = parse_test_report_counts(tr_text)
        if counts is None:
            errors.append("Could not parse passed/failed/skipped counts from reports/test_report.md")
        else:
            if counts["failed"] != 0:
                errors.append(f"reports/test_report.md reports {counts['failed']} failing tests")
            for label, doc_text in [("README.md", readme_text)] + (
                [("docs/results_details.md", rd_path.read_text(encoding="utf-8"))] if rd_path.exists() else []
            ):
                for tm in re.finditer(
                    r"`?(\d+)`?\s+(?:pytest\s+)?(?:tests?\s+)?passed\b", doc_text, flags=re.IGNORECASE
                ):
                    claimed = int(tm.group(1))
                    if claimed != counts["passed"]:
                        errors.append(
                            f"{label} claims '{tm.group(0)}' but reports/test_report.md has {counts['passed']} passed"
                        )

    # 6. Verify sensitivity headline numbers in docs/results_details.md if present
    if rd_path.exists():
        rd_text = rd_path.read_text(encoding="utf-8")
        sens_rows = summary.get("sample_size_sensitivity", {}).get("rows", [])
        if len(sens_rows) >= 2:
            sens_500 = next((r for r in sens_rows if r.get("Requested_Sample_Size") == 500000), None)
            sens_m = re.search(
                r"Macro-F1\s+`?([0-9.]+)\s*\[([0-9.]+),\s*([0-9.]+)\]`?\s+vs\s+`?([0-9.]+)\s*\[([0-9.]+),\s*([0-9.]+)\]`?",
                rd_text,
            )
            if sens_m and sens_500:
                _check_close(errors, "docs 500k Macro-F1", float(sens_m.group(1)), sens_500.get("F1_Macro"))
                _check_close(errors, "docs 500k CI low", float(sens_m.group(2)), sens_500.get("CI95_Low"))
                _check_close(errors, "docs 500k CI high", float(sens_m.group(3)), sens_500.get("CI95_High"))
                _check_close(
                    errors,
                    "docs 200k Macro-F1",
                    float(sens_m.group(4)),
                    sup.get("XGBoost", {}).get("f1_macro"),
                )

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify README numbers against generated evaluation/test reports."
    )
    parser.add_argument("--max-readme-lines", type=int, default=150)
    args = parser.parse_args()
    mismatches = check_readme_numbers(max_readme_lines=args.max_readme_lines)
    if mismatches:
        print("README number verification FAILED:")
        for err in mismatches:
            print(f"  - {err}")
        raise SystemExit(1)
    print(
        "README numbers and line budget verified cleanly against reports/evaluation_summary.json and reports/test_report.md."
    )


if __name__ == "__main__":
    main()
