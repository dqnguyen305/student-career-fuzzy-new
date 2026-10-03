import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "final_counseling_results.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "manfis_pseudo_labels.csv"
REQUIRED_COLUMNS = {
    "student_id",
    "student_name",
    "class",
    "top1_subject",
    "top1_score",
    "top2_subject",
    "top2_score",
    "suggested_combinations",
}
COMBINATION_PATTERN = re.compile(
    r"^([A-Z]{1,2}\d{2})\s*\(([^()]*)\)\s*:\s*([+-]?\d+(?:[.,]\d+)?)$"
)
MAX_COMBINATIONS = 4


def parse_combinations(value: str, row_number: int) -> list[dict[str, object]]:
    combinations = []
    for item in str(value).split("|"):
        match = COMBINATION_PATTERN.fullmatch(item.strip())
        if match is None:
            raise ValueError(f"Invalid combination format at CSV row {row_number}: {item!r}")

        subjects = [subject.strip() for subject in match.group(2).split(",")]
        if len(subjects) != 3 or any(not subject for subject in subjects):
            raise ValueError(f"Expected three subjects at CSV row {row_number}: {item!r}")

        combinations.append({
            "code": match.group(1).upper(),
            "subjects": ", ".join(subjects),
            "score": float(match.group(3).replace(",", ".")),
        })

    if not combinations or len(combinations) > MAX_COMBINATIONS:
        raise ValueError(
            f"Expected 1-{MAX_COMBINATIONS} combinations at CSV row {row_number}"
        )
    return combinations


def normalize_counseling_labels(
    input_path: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    input_path = Path(input_path)
    output_path = Path(output_path)
    source = pd.read_csv(input_path, dtype={"student_id": "string"})
    missing_columns = REQUIRED_COLUMNS - set(source.columns)
    if missing_columns:
        raise ValueError(f"Input file is missing columns: {sorted(missing_columns)}")

    source = source[list(REQUIRED_COLUMNS)].copy()
    for column in ("student_id", "student_name", "class", "top1_subject", "top2_subject"):
        source[column] = source[column].astype("string").str.strip()

    if source["student_id"].isna().any() or source["student_id"].eq("").any():
        raise ValueError("student_id must not be empty")
    if source["student_id"].duplicated().any():
        raise ValueError("student_id values must be unique")

    for column in ("top1_score", "top2_score"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
        values = source[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"{column} must contain finite values in [0, 1]")

    records = []
    for row_index, row in source.reset_index(drop=True).iterrows():
        row_number = row_index + 2
        combinations = parse_combinations(row["suggested_combinations"], row_number)
        record = {
            "student_id": row["student_id"],
            "student_name": row["student_name"],
            "class": row["class"],
            "top1_subject": row["top1_subject"].lower(),
            "top1_score": float(row["top1_score"]),
            "top2_subject": row["top2_subject"].lower(),
            "top2_score": float(row["top2_score"]),
            "label_source": "final_counseling_results_pseudo_label",
        }
        for combination_index in range(MAX_COMBINATIONS):
            prefix = f"combination_{combination_index + 1}"
            combination = combinations[combination_index] if combination_index < len(combinations) else {}
            record[f"{prefix}_code"] = combination.get("code", "")
            record[f"{prefix}_subjects"] = combination.get("subjects", "")
            record[f"{prefix}_score"] = combination.get("score", np.nan)
        records.append(record)

    normalized = pd.DataFrame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(output_path, index=False, encoding="utf-8-sig")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize counseling outputs into structured MANFIS pseudo-labels."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    normalized = normalize_counseling_labels(args.input, args.output)
    print(f"Normalized {len(normalized)} students to {args.output}")


if __name__ == "__main__":
    main()