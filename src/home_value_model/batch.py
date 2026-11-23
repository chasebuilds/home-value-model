"""Batch valuation of a CSV of subjects, with row-level validation and a one-page HTML report."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd

from .data import NUMERIC
from .features import DEFAULT_VALUATION_YEAR, REQUIRED
from .model import AVM

DISCLAIMER = (
    "This is a statistical estimate, not an appraisal. A USPAP-compliant appraisal "
    "requires a licensed or certified appraiser."
)
LIMITS = {
    "gr_liv_area": (200, 20000),
    "lot_area": (500, 2_000_000),
    "overall_qual": (1, 10),
    "overall_cond": (1, 9),
    "year_built": (1700, 2100),
}


def trust_text(band: str, flags: list[str]) -> str:
    """Plain-English guidance for one estimate."""
    lines = {
        "high": "This home looks like many homes the model has seen, and both models agree. "
        "Typical error on homes like this was small in testing, but it is still an estimate.",
        "medium": "Usable as a rough starting point. Check it against recent sales of similar "
        "homes before relying on it.",
        "low": "Do not rely on this number. The home is unusual, the two models disagree, or "
        "the data is thin for it. Use it only as a sanity check on a real valuation.",
    }
    text = lines[band]
    if flags:
        text += " Warnings: " + "; ".join(flags) + "."
    return text


def clean_subjects(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a subject table into rows that can be valued and rows that cannot.

    Returns (clean, problems). `clean` keeps the original index. `problems` has one row per
    skipped row with a reason. Fixable oddities (remodel year before build year, remodel year
    blank) are repaired and noted in a `notes` column on the clean rows.
    """
    df = raw.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing columns: {', '.join(missing_cols)}")
    notes: dict[Any, list[str]] = {i: [] for i in df.index}
    bad: dict[Any, str] = {}
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # A blank remodel year means never remodeled: use the build year.
    blank = df["year_remod"].isna() & df["year_built"].notna()
    for i in df.index[blank]:
        notes[i].append("remodel year blank, used year built")
    df.loc[blank, "year_remod"] = df.loc[blank, "year_built"]
    early = df["year_remod"] < df["year_built"]
    for i in df.index[early]:
        notes[i].append("remodel year before year built, used year built")
    df.loc[early, "year_remod"] = df.loc[early, "year_built"]
    for i in df.index:
        row = df.loc[i]
        empty = [c for c in REQUIRED if pd.isna(row[c]) or str(row[c]).strip() == ""]
        if empty:
            bad[i] = f"missing or non-numeric: {', '.join(empty)}"
            continue
        for c, (lo, hi) in LIMITS.items():
            if not lo <= row[c] <= hi:
                bad[i] = f"{c} = {row[c]:g} is outside {lo} to {hi}"
                break
    if "valuation_year" in df.columns:
        year = pd.to_numeric(df["valuation_year"], errors="coerce").fillna(DEFAULT_VALUATION_YEAR)
    else:
        year = pd.Series(DEFAULT_VALUATION_YEAR, index=df.index)
    df["yr_sold"] = year.astype(int)
    keep = df.index.difference(list(bad))
    clean = df.loc[keep, [*REQUIRED, "yr_sold"]].copy()
    clean["notes"] = ["; ".join(notes[i]) for i in keep]
    problems = pd.DataFrame({"row": list(bad), "reason": list(bad.values())})
    return clean, problems


def value_table(model: AVM, raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Value every valid row of `raw`. Returns (results, problems)."""
    clean, problems = clean_subjects(raw)
    if clean.empty:
        return pd.DataFrame(), problems
    pred = model.predict(clean)
    out = raw.loc[clean.index].copy()
    out["estimate"] = pred["estimate"]
    out["low"] = pred["low"]
    out["high"] = pred["high"]
    out["hedonic_estimate"] = pred["hedonic_estimate"]
    out["confidence_score"] = pred["confidence_score"]
    out["confidence_band"] = pred["confidence_band"]
    out["warnings"] = ["; ".join(f) for f in pred["flags"]]
    out["notes"] = clean["notes"]
    return out, problems


def html_report(model: AVM, subject: pd.DataFrame, year: int) -> str:
    """A standalone one-page analysis of one subject. No scripts, prints cleanly."""
    res = model.predict(subject, year).iloc[0]
    drivers = model.drivers(subject)
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(str(k))}</td><td>{esc(str(v))}</td></tr>" for k, v in subject.iloc[0].items()
    )
    drv = "".join(
        f"<tr><td>{esc(d['feature'])} vs a typical home</td><td>{d['effect_pct']:+.1f}%</td></tr>"
        for d in drivers
    )
    warn = "".join(f"<li>{esc(f)}</li>" for f in res["flags"]) or "<li>None</li>"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Home value analysis</title>
<style>
body{{font:15px/1.5 "Segoe UI",system-ui,sans-serif;max-width:720px;
margin:32px auto;padding:0 16px;color:#222}}
h1{{font-size:22px;margin:0 0 4px}} h2{{font-size:16px;margin:22px 0 6px}}
.big{{font-size:34px;font-weight:600}}
.box{{border:1px solid #bbb;padding:10px 14px;border-radius:8px}}
table{{border-collapse:collapse;width:100%}} td{{border-top:1px solid #ddd;padding:5px 6px}}
.note{{color:#555;font-size:13px}}
</style></head><body>
<h1>Home value analysis</h1>
<p class="box"><b>Not an appraisal.</b> {esc(DISCLAIMER)}</p>
<h2>Estimate</h2>
<div class="big">${res["estimate"]:,.0f}</div>
<div>80% range ${res["low"]:,.0f} to ${res["high"]:,.0f}.
Hedonic model alone: ${res["hedonic_estimate"]:,.0f}.</div>
<div>Confidence: {int(res["confidence_score"])}/100 ({esc(res["confidence_band"])})</div>
<p>{esc(trust_text(res["confidence_band"], res["flags"]))}</p>
<h2>Warnings</h2><ul>{warn}</ul>
<h2>Biggest price drivers (hedonic model)</h2><table>{drv}</table>
<h2>Subject inputs</h2><table>{rows}</table>
<p class="note">Valuation year {year}. Trained on {model.train_rows} sales,
{model.train_years[0]} to {model.train_years[1]}. Generated by home-value-model (Chase Builds).</p>
</body></html>
"""
