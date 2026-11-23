| Model | Form | *n*\* (95% interval) | First size, median | Ahead at 50 | 10th pct. gain at 10 / 20 / 50 | *τ*~b~ at 500 |
|:---|:---|:---:|:---:|---:|:---:|---:|
| Ridge | linear | 11.6 (8.7–14.8) | 20 | 83.3% | −0.161 / −0.054 / −0.006 | 0.967 |
| Ridge | concave | 11.3 (9.7–15.1) | 20 | 85.4% | −0.130 / −0.047 / −0.009 | 0.929 |
| Ridge | interaction | 13.5 (10.6–18.5) | 20 | 81.2% | −0.146 / −0.069 / −0.015 | 0.956 |
| Ridge | veto | none | never in 56% | 25.0% | −0.302 / −0.212 / −0.156 | 0.837 |
| WMin~*λ*~, learned | linear | 9.6 (6.8–14.0) | 20 | 81.2% | −0.131 / −0.071 / −0.022 | 0.967 |
| WMin~*λ*~, learned | concave | 8.9 (7.3–11.6) | 20 | 83.3% | −0.122 / −0.067 / −0.018 | 0.941 |
| WMin~*λ*~, learned | interaction | 11.9 (8.6–16.9) | 20 | 77.1% | −0.154 / −0.101 / −0.039 | 0.959 |
| WMin~*λ*~, learned | veto | 51.6 (37.9–79.6) | 50 (never in 17%) | 43.8% | −0.177 / −0.112 / −0.059 | 0.898 |

Pooled over the 144 compensatory scenarios, ridge regression first beat the rule at a median of 20 decisions (interquartile range 10–50, 90th percentile about 100) and was ahead after 50 decisions in 120 scenarios (67 of 72 without and 53 of 72 with label noise). The power law (7) fitted the mean ridge curves better than *a* − *b* exp(−*cn*) (AIC lower by 43–54). The break-even point of the mean curve is smaller than the median scenario-level break-even point because scenario-level gains are right-skewed: in scenarios with concentrated true weights the entropy-weighted rule is poor and the learned model overtakes it almost immediately, which pulls the mean curve up early.

