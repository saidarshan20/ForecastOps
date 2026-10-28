# Frozen experiment contract

Frozen before implementation and before inspecting test targets or metrics: 2026-10-28.

Source is UCI Bike Sharing `day.csv`; only `dteday` and `cnt` may enter modelling. Require 731 consecutive dates from 2011-01-01 through 2012-12-31. Chronological index partitions are initial train 0–590 (591 days), validation 591–660 (70 days), and untouched test 661–730 (70 days).

Use five nonoverlapping 14-day validation windows starting 2012-08-14, 2012-08-28, 2012-09-11, 2012-09-25, 2012-10-09. After model selection is recorded, use five test windows starting 2012-10-23, 2012-11-06, 2012-11-20, 2012-12-04, 2012-12-18. At each origin, fit on actuals strictly before the window; issue all 14 forecasts before revealing any window targets. Between windows, expand history using revealed actuals. Recursive regression and seasonal naive must use earlier predictions within each window.

Candidates and fixed parameters: seasonal naive (lag 7); last-value naive; Holt-Winters additive damped trend and additive weekly seasonality, estimated initialization; Ridge(StandardScaler, alpha=10); HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=42). Forecasts are clipped to zero and remain floating point. Model failures are recorded and make a model ineligible unless all validation windows succeed.

Selection is validation-only pooled MAE. Relative improvement = (seasonal-naive MAE − candidate MAE) / seasonal-naive MAE. Choose the lowest-MAE fully successful candidate only if it beats seasonal naive by at least 1%; otherwise choose seasonal naive. If baseline MAE is zero, retain baseline. Resolve exact MAE ties in this order: seasonal naive, last-value naive, Holt-Winters, Ridge, histogram gradient boosting. Freeze selected class and parameters before test evaluation; do not retune from test results.

Report pooled MAE, RMSE, WAPE (undefined if sum absolute actuals is zero), fold-level metrics, horizon-level MAE, and baseline-relative improvement. A separate monitoring experiment issues 70 one-day forecasts, one per test date, refitting the fixed selected class and baseline on actuals through the previous day before revealing that day's actual. Rolling MAE needs 28 paired days. Baseline-selected mode uses absolute drift against the initial 28-day MAE. The hosted artifact is fitted only through 2012-10-22 and predicts dates beginning 2012-10-23. This is a historical demo, not a live service.
