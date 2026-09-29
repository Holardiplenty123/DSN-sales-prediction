# DSN AI Bootcamp Qualification Hackathon: Product-Store Sales Prediction

## Overview

This project predicts `total\_sales` for each row in a hidden test set, using historical product-store sales data. It was built for the DSN AI Bootcamp qualification hackathon (ML track), where leaderboard performance and code quality both feed into bootcamp selection.

**Task type:** Regression
**Evaluation metric:** Root Mean Squared Error (RMSE), lower is better
**Final result:** CV RMSE \~1075.6 (stacked ensemble), 10th place on the leaderboard

## Files

|File|Purpose|
|-|-|
|`DSN\_Sales\_Prediction\_Final.ipynb`|Full notebook: EDA, cleaning, feature engineering, modeling, ensembling|
|`dsn\_sales\_prediction.py`|Script version of the same pipeline|
|`submission\_stacked.csv`|Final leaderboard submission|

## Problem Statement

Given product and store attributes (price, weight, category, store size, location tier, etc.), predict how much a given product sells for at a given store. Success is judged on how close the predictions are to actual sales, measured by RMSE.

## Data Exploration (EDA)

Before touching the data, the following were checked and visualized:

* **Target distribution**: `total\_sales` is right-skewed (mean \~2,175, range \~33-13,000), with a long tail of high-selling combinations. Confirmed with a histogram and boxplot.
* **Missing values**: `product\_weight\_kg` and `store\_size` both have substantial gaps, visualized with a bar chart.
* **Store effect**: average sales per store range from \~334 to \~3,660, a roughly 10x gap. This made `store\_code` one of the strongest candidate features.
* **Category effect**: average sales vary by `product\_category`, though less dramatically than by store.
* **Price relationship**: a scatter plot of `product\_price` vs `total\_sales` shows a clear positive relationship.
* **Categorical balance**: count plots of `fat\_content`, `store\_format`, and `store\_location\_tier` confirmed no severely rare categories.

## Data Cleaning

1. **Inconsistent text casing**: `product\_category` and `fat\_content` had the same values written in different cases (`Dairy`, `dairy`, `DAIRY`). Fixed by lowercasing and stripping whitespace, then merging near-duplicate labels (e.g. `'lf'` became `'low\_fat'`).
2. **`shelf\_visibility == 0`**: 422 rows had exactly 0, unrealistic for a product actually on a shelf. Treated as missing, then filled with the mean visibility for that product's category.
3. **`product\_weight\_kg` missing**: filled using the mean weight of the same `product\_code` first (a product should weigh the same everywhere), falling back to the category mean only if no other row of that product existed.
4. **`store\_size` missing**: each store has exactly one size, so missing values were filled by mapping from other rows of the same `store\_code`. Any store with no recorded size anywhere was labeled `'Unknown'`.
5. Train and test were combined before cleaning (target column excluded) so both received identical treatment, then split back apart.

This is more structure-aware than a generic median/mode fill. It borrows from rows that are *guaranteed* to share the true value (same product, same store) rather than guessing from the overall distribution.

## Outlier Handling

High-sales rows were checked with the IQR method. These were judged to be real, legitimate high-selling product/store combinations, not data errors, so no rows were deleted. Instead, a `log1p` transform was applied to the target during training (and reversed with `expm1` on predictions), which reduces the influence of the right skew without discarding any data.

## Feature Engineering

* `price\_per\_kg`: price normalized by product weight
* `price\_rank\_in\_category`: where a product's price sits (percentile) within its category
* `product\_count\_in\_train`: how many times a product appears in training data, as a proxy for how much history exists on it

## Encoding

Two strategies, matched to the model family:

* **Tree models (LightGBM, CatBoost)**: native categorical handling, columns kept as `category` dtype.
* **Linear/SVM models**: one-hot encoding for low-cardinality categoricals (`fat\_content`, `store\_size`, `store\_location\_tier`, `store\_format`), and **out-of-fold target encoding** for high-cardinality ones (`store\_code`, `product\_category`, `product\_code`, etc.).

Target encoding was computed strictly out-of-fold (each row's encoding built only from *other* folds' data) to avoid leaking target information into the feature. An in-fold version would inflate validation scores without a real leaderboard improvement.

## Multicollinearity Check

Correlation matrix and Variance Inflation Factor (VIF) were computed on the linear/SVM feature set. Two severe cases were found:

* `store\_code\_te` and `store\_format\_te` correlated at 0.99 (only 10 stores, so format is nearly determined by store identity), VIF over 100.
* `product\_price` and `price\_rank\_in\_category` correlated at 0.98 (rank is directly derived from price), VIF over 40.

Both redundant features were dropped from the linear feature set, bringing every VIF below 8. This step doesn't matter for tree models (which just split on whichever correlated feature is available) but matters a lot for linear model coefficient stability.

## PCA / Dimensionality Reduction

Tested, not assumed. After fixing multicollinearity, PCA (keeping 95% variance) reduced 19 features to 12, a modest reduction, since most of the real redundancy was already gone. A Ridge model trained on PCA components performed *worse* than one trained on the original (cleaned) features, confirming PCA wasn't needed here.

## Models Trained

|Model|Family|Notes|
|-|-|-|
|Ridge Regression|Linear, regularized|`RidgeCV`, cross-validated alpha|
|Ridge + PCA|Linear, regularized|Tested for comparison, performed worse|
|Random Forest|Bagged trees|400 trees, tuned depth/leaf size|
|SVR (RBF kernel)|Support vector machine|Scaled features required|
|LightGBM|Gradient boosting|Hyperparameters tuned via 40-trial random search|
|CatBoost|Gradient boosting|Hyperparameters tuned via random search|

All six were validated on identical 5-fold cross-validation splits, so their out-of-fold predictions could be fairly compared and combined.

## A Key Lesson: The CatBoost Overfitting Bug

An early version fed CatBoost the raw `product\_code` column (1,555 unique values across only 6,818 training rows, about 4.4 rows per product). Cross-validation RMSE dropped to an unbelievable 1024.6, but the actual leaderboard score came back at 1192.5, *worse* than the simplest baseline submission.

**Diagnosis:** removing the raw `product\_code` column (while keeping its out-of-fold target-encoded version) brought CV RMSE back to a believable 1106.6, consistent with the other models. The high-cardinality raw column had let CatBoost's internal categorical handling memorize near-duplicate rows within each fold instead of learning anything generalizable.

**Takeaway applied throughout the rest of the project:** an unusually large cross-validation improvement is a signal to investigate, not a result to celebrate.

## Ensembling: Weighted Blend vs. Stacking

Two combination strategies were tested on the six models' out-of-fold predictions:

1. **Weighted blend**: random search over blend weights (summing to 1) that minimize CV RMSE.
2. **Stacking**: a Ridge meta-model trained on the base models' out-of-fold predictions.

A first pass at stacking trained the meta-model on *all* the out-of-fold data and scored it on that same data, an in-sample estimate, not a genuine held-out one (the same mistake category as the CatBoost bug above). This was caught and fixed with a proper **nested cross-validation**: for each fold, the meta-model was trained only on the *other* folds' out-of-fold predictions. The honest estimate (1075.6) came out close to the naive one (1073.8), confirming this result was trustworthy rather than another false improvement.

|Approach|CV RMSE|
|-|-|
|Best single model (CatBoost)|1106.6|
|Weighted blend|1098.9|
|Stacked ensemble (honest, nested CV)|**1075.6**|

## Final Model

The stacking ensemble (Ridge meta-model over Ridge, Random Forest, SVR, LightGBM, and CatBoost) was used for the final submission.

## Leaderboard Progress

|Version|Change|CV RMSE|Leaderboard|
|-|-|-|-|
|v1|Baseline LightGBM|1124.3|1123.40|
|v2|+ out-of-fold target encoding|1118.5|1123.36|
|v3|+ interaction features + CatBoost|1024.6 (false)|1192.50 (overfit)|
|v5|+ LightGBM hyperparameter search|1114.9|1121.30|
|v6|Fixed CatBoost (leak removed)|1106.6|1110.96|
|v7|3-way ensemble (tied v6)|1106.6|1110.96|
|**Final**|**Stacked ensemble, 6 models**|**1075.6**|n/a|

**Final leaderboard rank: 10th.**



## Summary

The strongest individual predictors were store identity and product price, both confirmed by EDA (bar charts, scatter plot) and later by feature importance. The project's real value wasn't a single clever feature, but a disciplined process: structure-aware imputation instead of generic fills, an explicit multicollinearity check instead of assuming it away, PCA tested rather than assumed helpful, and, twice, an unusually good validation score treated as a bug to hunt down rather than a result to trust blindly. Both times, the honest number was found and used instead of the optimistic one.

