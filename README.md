# Monte Carlo comparison of ranking methods for decision support systems

Code, seeds, raw results and supplementary material for the paper
"Choosing the Ranking Method for Decision Support Systems: A Monte Carlo Comparison of
Data-Driven Fuzzy Inference, Linear Aggregation and Machine Learning" (Kerimkhulle S., Kakim A.; submitted).

## Contents

- `src/` — experiment code. `fuzzy_priority/` is the data-driven Mamdani engine used in the paper
  (membership functions, FCM partitions, entropy weights, monotone rule generation).
- `data/` — the five PMLB datasets (ESL, LEV, SWD, ERA, Car Evaluation), copied from PMLB.
- `results/` — raw per-problem results as gzip-compressed CSV (one row per problem and method; pandas reads `.csv.gz` directly) and analysis reports.
- `supplement/Supplementary_Material.pdf` — implementation details and additional results (S1–S8).

## Environment

Python 3.13, NumPy 2.5, SciPy 1.18, pandas 3.0, scikit-learn 1.9, joblib 1.6, statsmodels 0.15;
plotly 6 with kaleido 1.x for the figures.

## Reproduce

```
cd src
python simulate.py        ../results/full_R30.csv.gz 30 2   # main experiment, 384 scenarios x 30 replications
python simulate2.py       ../results/abl_R30.csv.gz  30 2   # fuzzy ablations, TSK, Choquet-type, monotone GBM, expert weights
python simulate3.py       ../results/r3_R30.csv.gz   30 2   # learned WMin, lambda sweep, ties, monotonicity check
python simulate4.py       ../results/r4_R30.csv.gz   30 2   # regularisation control (non-negative ridge), veto-penalty frequency, FCM fallback
python simulate_shape.py  ../results/shape_R10.csv.gz 10 2  # theta, beta (veto) and gamma (interaction) sensitivity
python simulate_curve.py  ../results/curve_R30.csv.gz 30    # learning curves of ridge regression, n = 10..500
python simulate_curve2.py ../results/curve_wmin_R30.csv.gz 30  # learning curves of the learned WMin
python bench3.py ../data ../results/bench3_R20.csv.gz 20    # five public datasets, 20 splits, tau_b and Somers' D
python merge_all3.py ../results                       # builds all3_R30.csv.gz
python analyze3.py ../results/all3_R30.csv.gz ../results/r3_R30.csv.gz ../results/curve_R30.csv.gz \
       ../results/curve_wmin_R30.csv.gz ../results/shape_R10.csv.gz ../results/bench3_R20.csv.gz > ../results/report3.txt
```

`all3_R30.csv.gz` (built by `merge_all3.py`) is the union of `full_R30.csv.gz`, `abl_R30.csv.gz` and `r3_R30.csv.gz` (methods that appear in several
files are identical, which `analyze2.py` and `analyze3.py` check).

Seeds are fixed: problem `r` of scenario `s` uses `numpy.random.default_rng(2026 + 1000*s + r)`, so all results
are reproduced exactly on the stated library versions.

## Licence and citation

Code: MIT licence (see `LICENSE`). The PMLB datasets in `data/` keep their original licences. If you use this code, please cite the paper (see `CITATION.cff`).
