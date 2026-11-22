"""Command line: avm train | predict | synth | fetch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .data import NUMERIC, fetch_ames, load_ames, synthetic, time_split
from .features import DEFAULT_VALUATION_YEAR, REQUIRED
from .model import AVM
from .report import evaluate, page_model

DISCLAIMER = (
    "This is a statistical estimate, not an appraisal. A USPAP-compliant appraisal "
    "requires a licensed or certified appraiser."
)
DEFAULT_MODEL = Path("models/avm.joblib")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def trust_text(row: pd.Series) -> str:
    """Plain-English guidance for one estimate."""
    band = row["confidence_band"]
    lines = {
        "high": "This home looks like many homes the model has seen, and both models agree. "
        "Typical error on homes like this was small in testing, but it is still an estimate.",
        "medium": "Usable as a rough starting point. Check it against recent sales of similar "
        "homes before relying on it.",
        "low": "Do not rely on this number. The home is unusual, the two models disagree, or "
        "the data is thin for it. Use it only as a sanity check on a real valuation.",
    }
    text = lines[band]
    if row["flags"]:
        text += " Warnings: " + "; ".join(row["flags"]) + "."
    return text


def read_subject(path: Path) -> tuple[pd.DataFrame, int]:
    raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED if k not in raw]
    if missing:
        raise SystemExit(f"subject is missing fields: {', '.join(missing)}")
    year = int(raw.pop("valuation_year", DEFAULT_VALUATION_YEAR))
    frame = pd.DataFrame([{k: raw[k] for k in REQUIRED}])
    for c in NUMERIC:
        frame[c] = pd.to_numeric(frame[c])
    return frame, year


def cmd_train(args: argparse.Namespace) -> int:
    if args.source == "ames":
        df = load_ames(Path(args.data) if args.data else None)
    else:
        df = synthetic(args.n, args.seed)
    train, test = time_split(df, args.test_from)
    held_out = AVM(args.seed).fit(train)
    model = AVM(args.seed).fit(df) if args.all_data else held_out
    rep = evaluate(held_out, test, train)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    if args.report:
        Path(args.report).write_text(json.dumps(rep, indent=2), encoding="utf-8")
    if args.page_model:
        Path(args.page_model).write_text(json.dumps(page_model(model), indent=1), encoding="utf-8")
    for name in ("baseline_ppsf", "hedonic", "gbm", "blend"):
        m = rep[name]
        print(
            f"{name:8s} MdAPE {_pct(m['mdape'])}  PPE10 {_pct(m['ppe10'])}  "
            f"FSD {m['fsd']:.3f}  COD {m['cod']:.1f}  (n={int(m['n'])})"
        )
    print(f"saved {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    path = Path(args.model)
    if not path.exists():
        raise SystemExit(f"no model at {path}. Run: avm train --source synthetic (or ames)")
    model = AVM.load(path)
    subject, year = read_subject(Path(args.subject))
    res = model.predict(subject, year).iloc[0]
    drivers = model.drivers(subject)
    if args.json:
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}
        payload["drivers"] = drivers
        payload["trust"] = trust_text(res)
        payload["disclaimer"] = DISCLAIMER
        print(json.dumps(payload, indent=2))
        return 0
    print(
        f"Estimate: ${res['estimate']:,.0f}   (80% range ${res['low']:,.0f} to ${res['high']:,.0f})"
    )
    print(f"Hedonic model alone: ${res['hedonic_estimate']:,.0f}")
    print(f"Confidence: {res['confidence_score']}/100 ({res['confidence_band']})")
    print("Biggest price drivers vs a typical home (hedonic model):")
    for d in drivers:
        print(f"  {d['feature']:22s} {d['effect_pct']:+.1f}%")
    print(trust_text(res))
    print(DISCLAIMER)
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    synthetic(args.n, args.seed).to_csv(args.out, index=False)
    print(f"wrote {args.n} synthetic rows to {args.out}")
    return 0


def cmd_fetch(_: argparse.Namespace) -> int:
    print(fetch_ames())
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="avm", description=DISCLAIMER)
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="fit on a training period, test on later sales")
    t.add_argument("--source", choices=["ames", "synthetic"], default="ames")
    t.add_argument("--data", help="path to AmesHousing.txt (default: download to the user cache)")
    t.add_argument("--n", type=int, default=1500, help="synthetic rows")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--test-from", type=int, default=2009, help="first sale year in the test set")
    t.add_argument("--all-data", action="store_true", help="refit on all years for use")
    t.add_argument("--out", default=str(DEFAULT_MODEL))
    t.add_argument("--report", help="write the evaluation as JSON")
    t.add_argument("--page-model", help="write hedonic coefficients for the static page")
    t.set_defaults(func=cmd_train)

    q = sub.add_parser("predict", help="estimate one subject property from a JSON file")
    q.add_argument("subject")
    q.add_argument("--model", default=str(DEFAULT_MODEL))
    q.add_argument("--json", action="store_true")
    q.set_defaults(func=cmd_predict)

    s = sub.add_parser("synth", help="write a seeded synthetic data set")
    s.add_argument("--n", type=int, default=600)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", default="data/sample_synthetic.csv")
    s.set_defaults(func=cmd_synth)

    f = sub.add_parser("fetch", help="download the Ames file to the user cache")
    f.set_defaults(func=cmd_fetch)

    args = p.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
