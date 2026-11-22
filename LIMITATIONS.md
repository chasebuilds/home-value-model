# Limitations

This is an analysis and education tool. It is not an appraisal. A USPAP-compliant appraisal requires a licensed or certified appraiser.

## Data
- The only real data is Ames, Iowa sales from 2006 to 2010 (single-family, normal sales, 2,001 after filtering). Estimates are in 2010 Ames dollars. Using them for another town or another year is wrong.
- The test years (2009 and 2010) are a mild test of time drift. The model has no market-level input, so a market that moves fast will push errors up and bias them one way. Both models already run about 1% low on the test sales.
- Ames is a college town with one dominant employer. Do not assume the numbers generalize to rural or Eastern Kentucky markets.
- The Ames file has no license text I could confirm for redistribution, so it is downloaded at run time and never stored in the repo. It is the assessor's view of each sale, not an MLS record.

## Model
- Thirteen inputs only. It cannot see condition detail, finishes, views, lot shape, flood zone, schools, or anything not in the form.
- With about 1,300 training homes, gradient boosting is no better than the linear model. Do not read this as a verdict on gradient boosting in general.
- The hedonic coefficients are associations, not causes. Adding a bathroom will not raise a price by the coefficient.
- The 80% range comes from past out-of-fold errors. It was close to 80% on test overall (79.4%) but slightly too narrow for high-confidence homes (76.5%).
- The confidence score ranks risk. It is not a probability.
- The static demo page runs only the hedonic model and its range. The gradient-boosted model, confidence score and warnings need the CLI.

## Use
- Do not use an estimate as the basis for a loan, a tax assessment, a listing price or a legal dispute.
- Synthetic data in `data/` and in tests is made up. It says nothing about real prices.
