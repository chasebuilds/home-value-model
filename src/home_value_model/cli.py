"""Command line: avm train | predict | batch | report | synth | fetch."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .batch import DISCLAIMER, html_report, trust_text, value_table
from .data import NUMERIC, fetch_ames, load_ames, synthetic, time_split
from .features import DEFAULT_VALUATION_YEAR, REQUIRED
from .model import AVM
from .report import evaluate, page_model

DEFAULT_MODEL = Path("models/avm.joblib")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


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
    model = _load_model(args.model)
    subject, year = read_subject(Path(args.subject))
    res = model.predict(subject, year).iloc[0]
    drivers = model.drivers(subject)
    if args.json:
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in res.items()}
        payload["drivers"] = drivers
        payload["trust"] = trust_text(res["confidence_band"], res["flags"])
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
    print(trust_text(res["confidence_band"], res["flags"]))
    print(DISCLAIMER)
    return 0


def _load_model(path_str: str) -> AVM:
    path = Path(path_str)
    if not path.exists():
        raise SystemExit(f"no model at {path}. Run: avm train --source synthetic (or ames)")
    return AVM.load(path)


def cmd_batch(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    raw = pd.read_csv(args.input, dtype=str)
    try:
        results, problems = value_table(model, raw)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    out = Path(args.out)
    results.to_csv(out, index=False)
    print(f"valued {len(results)} of {len(raw)} rows, wrote {out}")
    if len(problems):
        prob_path = out.with_name(out.stem + "_skipped.csv")
        problems.to_csv(prob_path, index=False)
        print(f"skipped {len(problems)} rows, reasons in {prob_path}")
    print(DISCLAIMER)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    model = _load_model(args.model)
    subject, year = read_subject(Path(args.subject))
    Path(args.out).write_text(html_report(model, subject, year), encoding="utf-8")
    print(f"wrote {args.out}")
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

    b = sub.add_parser("batch", help="value every row of a CSV of subject properties")
    b.add_argument("input")
    b.add_argument("--model", default=str(DEFAULT_MODEL))
    b.add_argument("--out", default="estimates.csv")
    b.set_defaults(func=cmd_batch)

    r = sub.add_parser("report", help="write a one-page HTML analysis of one subject")
    r.add_argument("subject")
    r.add_argument("--model", default=str(DEFAULT_MODEL))
    r.add_argument("--out", default="report.html")
    r.set_defaults(func=cmd_report)

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
