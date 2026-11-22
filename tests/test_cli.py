import json
from pathlib import Path

import pytest

from home_value_model.cli import main

SUBJECT = {
    "gr_liv_area": 1500,
    "lot_area": 9000,
    "overall_qual": 6,
    "overall_cond": 5,
    "year_built": 1990,
    "year_remod": 2000,
    "total_bsmt_sf": 800,
    "full_bath": 2,
    "half_bath": 1,
    "bedrooms": 3,
    "garage_cars": 2,
    "fireplaces": 1,
    "neighborhood": "ZoneD",
    "valuation_year": 2010,
}


def _train(tmp_path: Path, n: int = 500) -> Path:
    model = tmp_path / "m.joblib"
    args = ["train", "--source", "synthetic", "--n", str(n), "--out", str(model)]
    assert main([*args, "--report", str(tmp_path / "r.json")]) == 0
    return model


def test_train_then_predict(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    model = _train(tmp_path)
    assert json.loads((tmp_path / "r.json").read_text())["gbm"]["ppe10"] > 0.4
    subject = tmp_path / "s.json"
    subject.write_text(json.dumps(SUBJECT))
    capsys.readouterr()
    assert main(["predict", str(subject), "--model", str(model), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["estimate"] > 50_000 and "USPAP" in out["disclaimer"] and out["trust"]
    assert main(["predict", str(subject), "--model", str(model)]) == 0
    assert "not an appraisal" in capsys.readouterr().out


def test_predict_without_model_fails_clearly(tmp_path: Path) -> None:
    subject = tmp_path / "s.json"
    subject.write_text(json.dumps(SUBJECT))
    with pytest.raises(SystemExit, match="avm train"):
        main(["predict", str(subject), "--model", str(tmp_path / "nope.joblib")])


def test_missing_field_is_reported(tmp_path: Path) -> None:
    model = _train(tmp_path, 400)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"gr_liv_area": 1000}))
    with pytest.raises(SystemExit, match="missing fields"):
        main(["predict", str(bad), "--model", str(model)])


def test_synth_command(tmp_path: Path) -> None:
    out = tmp_path / "s.csv"
    assert main(["synth", "--n", "50", "--out", str(out)]) == 0
    assert len(out.read_text().splitlines()) == 51
