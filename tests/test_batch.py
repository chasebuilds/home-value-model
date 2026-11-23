import csv
from pathlib import Path

import pandas as pd
import pytest

from home_value_model.batch import clean_subjects, html_report, trust_text, value_table
from home_value_model.cli import main
from home_value_model.model import AVM

GOOD = {
    "gr_liv_area": "1500",
    "lot_area": "9000",
    "overall_qual": "6",
    "overall_cond": "5",
    "year_built": "1990",
    "year_remod": "2000",
    "total_bsmt_sf": "800",
    "full_bath": "2",
    "half_bath": "1",
    "bedrooms": "3",
    "garage_cars": "2",
    "fireplaces": "1",
    "neighborhood": "ZoneD",
}


def _table(*overrides: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame([{**GOOD, **o} for o in overrides], dtype=str)


def test_clean_rows_pass() -> None:
    clean, problems = clean_subjects(_table({}))
    assert len(clean) == 1 and problems.empty
    assert clean["yr_sold"].iloc[0] == 2010


def test_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        clean_subjects(pd.DataFrame({"gr_liv_area": ["1000"]}))


def test_header_spelling_is_forgiven() -> None:
    raw = _table({}).rename(columns={"gr_liv_area": " Gr Liv Area "})
    clean, _ = clean_subjects(raw)
    assert len(clean) == 1


def test_bad_rows_are_skipped_with_reasons() -> None:
    raw = _table({}, {"gr_liv_area": "abc"}, {"overall_qual": "11"}, {"neighborhood": ""})
    clean, problems = clean_subjects(raw)
    assert list(clean.index) == [0]
    reasons = " | ".join(problems["reason"])
    assert "gr_liv_area" in reasons and "overall_qual" in reasons and "neighborhood" in reasons


def test_remodel_year_repairs_are_noted() -> None:
    raw = _table({"year_remod": "1980"}, {"year_remod": ""})
    clean, problems = clean_subjects(raw)
    assert problems.empty
    assert clean["year_remod"].tolist() == [1990, 1990]
    assert all(clean["notes"])


def test_per_row_valuation_year() -> None:
    raw = _table({"valuation_year": "2008"}, {"valuation_year": ""})
    clean, _ = clean_subjects(raw)
    assert clean["yr_sold"].tolist() == [2008, 2010]


def test_value_table_keeps_original_columns(fitted: AVM) -> None:
    raw = _table({}, {"gr_liv_area": "abc"}, {"gr_liv_area": "2400"})
    results, problems = value_table(fitted, raw)
    assert list(results.index) == [0, 2] and len(problems) == 1
    assert {"estimate", "low", "high", "confidence_band", "warnings"} <= set(results.columns)
    assert results["estimate"].iloc[1] > results["estimate"].iloc[0]


def test_value_table_all_bad_rows(fitted: AVM) -> None:
    results, problems = value_table(fitted, _table({"gr_liv_area": "x"}))
    assert results.empty and len(problems) == 1


def test_trust_text_bands() -> None:
    assert "Do not rely" in trust_text("low", [])
    assert "Warnings: a; b." in trust_text("high", ["a", "b"])


def test_html_report_contents(fitted: AVM) -> None:
    subject = _table({}).drop(columns=[]).copy()
    for c in subject.columns:
        if c != "neighborhood":
            subject[c] = pd.to_numeric(subject[c])
    page = html_report(fitted, subject, 2010)
    assert "Not an appraisal" in page and "USPAP" in page and "<script" not in page
    assert "ZoneD" in page and "80% range" in page


def test_batch_and_report_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = tmp_path / "m.joblib"
    assert main(["train", "--source", "synthetic", "--n", "500", "--out", str(model)]) == 0
    src = tmp_path / "in.csv"
    with src.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(GOOD))
        w.writeheader()
        w.writerow(GOOD)
        w.writerow({**GOOD, "gr_liv_area": "n/a"})
    out = tmp_path / "out.csv"
    assert main(["batch", str(src), "--model", str(model), "--out", str(out)]) == 0
    assert "valued 1 of 2" in capsys.readouterr().out
    assert len(pd.read_csv(out)) == 1
    skipped = pd.read_csv(tmp_path / "out_skipped.csv")
    assert "gr_liv_area" in skipped["reason"].iloc[0]
    subject = tmp_path / "s.json"
    subject.write_text(
        '{"valuation_year": 2010, '
        + ", ".join(
            f'"{k}": {v if k != "neighborhood" else chr(34) + v + chr(34)}' for k, v in GOOD.items()
        )
        + "}"
    )
    page = tmp_path / "r.html"
    assert main(["report", str(subject), "--model", str(model), "--out", str(page)]) == 0
    assert "Home value analysis" in page.read_text(encoding="utf-8")


def test_batch_bad_columns_exit(tmp_path: Path) -> None:
    model = tmp_path / "m.joblib"
    main(["train", "--source", "synthetic", "--n", "400", "--out", str(model)])
    src = tmp_path / "in.csv"
    src.write_text("a,b\n1,2\n")
    with pytest.raises(SystemExit, match="missing columns"):
        main(["batch", str(src), "--model", str(model)])
