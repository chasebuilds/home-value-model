# home-value-model

[![ci](https://github.com/chasebuilds/home-value-model/actions/workflows/ci.yml/badge.svg)](https://github.com/chasebuilds/home-value-model/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An automated valuation model (AVM) done honestly: two models, a time-based test on later sales, the standard AVM accuracy metrics, and a confidence score that says when not to trust an estimate.

> **Not an appraisal.** This is an analysis and education tool. A USPAP-compliant appraisal requires a licensed or certified appraiser.

**Live demo:** <https://chasebuilds.github.io/home-value-model/> (the hedonic model running in your browser)

![Screenshot of the demo page](docs/screenshot.jpg)

## What it does

- Fits a **hedonic log-price regression** (every coefficient readable) and a **gradient-boosted model** (scikit-learn `HistGradientBoostingRegressor`) on the Ames, Iowa housing sales.
- Trains on sales from 2006 to 2008 and tests on 2009 and 2010 sales it never saw. No random shuffle across time.
- Scores the result with AVM metrics: median absolute % error (MdAPE), PPE10 (share of estimates within 10% of the sale price), forecast standard deviation (FSD), plus COD and PRD.
- Gives each estimate a **confidence score (0 to 100)**, a plain-English band (low, medium, high), an 80% value range, and warnings.
- Explains the model with permutation feature importance and per-home price drivers.

## Quick start

```powershell
py -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\avm train --source ames --report results.json   # downloads Ames to your cache
.venv\Scripts\avm predict examples/subject.json
```

Without network, use the seeded synthetic set instead: `avm train --source synthetic`.

Example output (`avm predict examples/subject.json`, model refit on all 2006 to 2010 sales):

```
Estimate: $188,200   (80% range $172,600 to $205,300)
Hedonic model alone: $190,700
Confidence: 91/100 (high)
Biggest price drivers vs a typical home (hedonic model):
  age                    +8.8%
  living area            +5.7%
  neighborhood           -1.6%
  ...
```

`avm predict` also takes `--json`. The subject file lists 12 numeric fields and a neighborhood (see [examples/subject.json](examples/subject.json)); `valuation_year` is optional and sets the age basis.

## Model card

**Data.** Ames, Iowa Assessor's Office residential sales, 2006 to 2010, as published by Dean De Cock, "Ames, Iowa: Alternative to the Boston Housing Data Set", *Journal of Statistics Education* 19(3), 2011 ([paper](https://jse.amstat.org/v19n3/decock.pdf), [data](https://jse.amstat.org/v19n3/decock/AmesHousing.txt)). I could not find a license that clearly allows redistribution, so **the data is not in this repo**: `avm train --source ames` downloads it to your user cache at run time. The parcel ID column is dropped on load. Following the author's advice, only normal-condition single-family sales under 4,000 sq ft of living area are kept: 2,001 sales (1,283 train, 718 test).

**Inputs (13).** Living area, lot area, overall quality, overall condition, year built, year remodeled, basement area, full and half baths, bedrooms, garage spaces, fireplaces, neighborhood.

**Split.** Train: sales in 2006 to 2008. Test: sales in 2009 and 2010. The confidence layer and ranges are built only from out-of-fold errors inside the training years.

**Results on the 718 held-out 2009 to 2010 sales** (reproduce with `avm train --source ames --report results.json`; the full output is in [docs/results_ames.json](docs/results_ames.json)):

| Model | MdAPE | PPE10 | PPE5 | FSD | COD | Median ratio |
|---|---|---|---|---|---|---|
| Rule of thumb (neighborhood median $/sq ft) | 11.6% | 43.0% | 21.4% | 0.199 | 15.9 | 0.986 |
| Hedonic (log-price ridge) | 6.2% | 72.3% | 42.6% | 0.106 | 8.1 | 0.989 |
| Gradient boosting | 6.0% | 72.4% | 43.2% | 0.108 | 8.0 | 0.988 |
| Average of the two (log scale) | 5.6% | 75.5% | 44.8% | 0.100 | 7.6 | 0.986 |

- The two models are about equally accurate here. Boosting is not clearly better than the linear model on 1,283 training homes.
- The default estimate in the CLI is the gradient-boosted one. The average of the two scored best on test, but I picked the default before seeing the test numbers, so I report the average and did not switch to it.
- Median ratio is estimate / sale price: both models run about 1% low on later sales. The 2008 to 2010 market moved, and the models do not know the calendar year.
- FSD here is the standard deviation of sale price / estimate.

**Confidence score.** A small error model predicts how wrong the estimate is likely to be for this home, using the home's features and how far the two models disagree. Homes outside the central 98% of training values, or in an unseen neighborhood, are pushed lower. The score is the share of training homes that looked riskier, so 80 means "safer than about 80% of training homes". Bands: high 67 and up, medium 33 to 66, low under 33. Held-out check, using the gradient-boosted estimate:

| Confidence band | Test homes | MdAPE | PPE10 | Sale inside 80% range |
|---|---|---|---|---|
| high | 213 | 4.8% | 86.4% | 76.5% |
| medium | 221 | 5.8% | 73.8% | 78.7% |
| low | 284 | 7.7% | 61.6% | 82.0% |
| all | 718 | 6.0% | 72.4% | 79.4% (target 80%) |

The score does what it should: higher bands were more accurate on later sales. The ranges are close to their 80% target overall, a little too narrow for the high band.

**What drives the estimate.** Permutation importance on the test sales (rise in MdAPE, in percentage points, when one input is shuffled):

| Input | Gradient boosting | Hedonic |
|---|---|---|
| Living area | +6.4 | +8.3 |
| Overall quality | +3.4 | +3.8 |
| Year built | +3.0 | +6.0 |
| Basement area | +1.9 | +1.2 |
| Overall condition | +1.3 | +2.6 |
| Neighborhood | +0.9 | +1.4 |

Everything else was under one point. Hedonic coefficients are in [docs/model.json](docs/model.json) (log-price per unit; for example the living-area elasticity is about 0.42).

## When to trust an estimate, in plain English

**Trust it more when** the home is ordinary for its area: typical size, age and quality, in a neighborhood the model has seen many sales from, and the two models land within a few percent of each other. Confidence is "high" and there are no warnings.

**Trust it less, or not at all, when:**
- the home is unusual (very large, very old, no basement, luxury finishes, odd layout);
- the model warns the neighborhood was never seen or an input is outside the usual range;
- the two models disagree by more than 15%;
- the market has moved since 2010 (it has; this model only knows 2006 to 2010 Ames);
- condition matters a lot and the data cannot see it (a gutted kitchen, a failing roof, a recent remodel not reflected in year remodeled);
- the home is outside Ames, Iowa. Do not use these prices anywhere else.

An estimate is a starting point for a human who knows the market. It never replaces comparable sales, an inspection or an appraisal. See [LIMITATIONS.md](LIMITATIONS.md).

## Layout

```
src/home_value_model/   data.py features.py model.py metrics.py report.py cli.py
tests/                  22 tests (metrics, split, models, confidence, CLI, loader)
data/sample_synthetic.csv   600 seeded fake homes, used by tests and CI (no network)
docs/                   demo page, hedonic coefficients, saved results, screenshot
examples/subject.json   a sample subject property
```

## Development

CI runs ruff, mypy (strict) and pytest on Python 3.11 and 3.13. It uses only the committed synthetic sample, so it never touches the network.

```powershell
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mypy
.venv\Scripts\python -m pytest
```

The synthetic data (`avm synth --n 600 --seed 0`) comes from a made-up price formula with noise. It exists to test the code, not to say anything about real prices.

## License

MIT, see [LICENSE](LICENSE). The Ames data is not included and keeps its own terms; cite De Cock (2011) if you use it.

Built by Chase Builds.
