# Final Essay Report: Sales Forecasting with Machine Learning

**Course:** Introduction to Machine Learning  
**Semester:** 2nd Semester, Academic Year 2025-2026  
**Student:** K234131613  
**Dataset:** `K234131613.csv`

---

## Abstract

This report presents an end-to-end machine learning workflow for retail sales forecasting using the dataset `K234131613.csv`. The study includes exploratory data analysis, feature engineering, categorical encoding strategies, and model development with Support Vector Regression (SVR) and Decision Tree Regression. Experimental results show that Decision Tree Regression provides stronger predictive performance than SVR on this dataset, while also supporting straightforward rule-based interpretation for practical deployment.

---

## 1. Dataset Introduction

### 1.1 Variables and Practical Meaning

The dataset contains transactional and store/item descriptors:

- `date`: transaction date  
- `store_nbr`: store identifier  
- `item_nbr`: item identifier  
- `type`: store type category (B/C in this subset)  
- `cluster`: store cluster index  
- `state`: store location (state/province)  
- `family`: product family  
- `class`: product class  
- `perishable`: perishable indicator (0/1)  
- `store_type`: encoded store type  
- `family_encoded`, `state_encoded`: encoded categorical labels  
- `unit_sales`: target variable (units sold)

### 1.2 Data Size and Types

- Number of observations: **56,755**
- Number of columns: **20** (after deriving date-based and engineered columns during processing)
- Missing values: **0**
- Duplicate rows: **0**

### 1.3 Descriptive Statistics of Key Numerical Variables

| Variable | Mean | Median | Std. Dev. | Min | Max |
|---|---:|---:|---:|---:|---:|
| `unit_sales` | 7.1443 | 4.0000 | 15.5913 | 0.0 | 1457.0 |
| `cluster` | 8.3760 | 6.0000 | 4.9699 | 3.0 | 16.0 |
| `store_nbr` | 23.3215 | 18.0000 | 12.1971 | 9.0 | 54.0 |
| `item_nbr` | 611,310.3629 | 583,978.0000 | 304,236.4430 | 103,501.0 | 1,118,683.0 |
| `class` | 1,875.6004 | 1,122.0000 | 1,064.5661 | 1,002.0 | 7,034.0 |

---

## 2. Problem Statement and Workflow

### 2.1 Task Definition

This is a **supervised regression** problem: predict continuous `unit_sales` from item-, store-, and time-related predictors.

### 2.2 Selected Models and Motivation

- **Support Vector Regression (SVR)**: strong nonlinear modeling via kernel methods, robust under high-dimensional transformed features.
- **Decision Tree Regressor**: interpretable nonlinear model, handles mixed feature interactions effectively.

### 2.3 Workflow Summary

1. Load and validate dataset quality  
2. Exploratory analysis and visualization  
3. Outlier analysis on `unit_sales`  
4. Feature engineering and preprocessing  
5. Train/test split  
6. SVR hyperparameter experiments  
7. Decision Tree hyperparameter experiments  
8. Model evaluation and deployment recommendation

---

## 3. Data Processing and Exploratory Analysis

### 3.1 Data Cleaning and Consistency Checks

- Missing-value handling: not required (no missing values found)
- Duplicate removal: not required (0 duplicates)
- Type consistency: numeric casting verified for key numerical variables
- Date conversion: `date` converted to datetime format

### 3.2 Visual Analytics and Pattern Discovery

Two required pattern-focused visual insights:

1. **Sales by store type (`type` / `store_type`)**  
   - Total sales by type:  
     - Type C: **229,356.200**  
     - Type B: **176,118.131**  
   - Interpretation: Type C contributes a larger share of total sales.

2. **Daily sales trend over time**  
   - Time range: **2013-01-05** to **2013-02-12**  
   - Example values: first day = **131,064.697**, last day = **81,034.605**  
   - Interpretation: aggregate daily sales show substantial fluctuations; temporal information is relevant for forecasting.

### 3.3 Outlier Detection and Modeling Impact

Using the IQR rule on `unit_sales`:

- Q1 and Q3 produce bounds: **[-7.0, 17.0]**
- Outlier count: **4,739** observations (**8.35%**)

Potential impact:

- High positive outliers may inflate RMSE and bias models toward rare peaks.
- Tree-based models may fit local spikes if unconstrained.
- SVR may underfit extreme spikes depending on `C` and `epsilon`.
- A robust extension could include log-transforming target or robust loss alternatives.

---

## 4. Feature Engineering (Question 2)

### 4.1 Engineered Features

At least four meaningful derived features were constructed:

1. `year` (from `date`)  
2. `month` (from `date`)  
3. `day` (from `date`)  
4. `is_weekend` (binary indicator from day-of-week)  
5. `item_mean_sales` = mean historical sales per `item_nbr`  
6. `store_mean_sales` = mean historical sales per `store_nbr`  
7. `family_mean_sales` = mean historical sales per `family`

### 4.2 Modeling Rationale per Feature

- **Calendar features (`year`, `month`, `day`)**: capture temporal seasonality and periodic demand variation.
- **`is_weekend`**: represents behavioral shifts in customer shopping patterns.
- **`item_mean_sales`**: encodes product-level baseline demand intensity.
- **`store_mean_sales`**: encodes store-specific demand scale and local customer volume.
- **`family_mean_sales`**: captures category-level demand priors and substitution effects.

### 4.3 Bias-Variance Implications

Feature engineering can reduce **bias** by exposing latent structure (seasonality, product/store effects) and can increase **variance** when many correlated or noisy engineered variables are added. In this report, aggregated mean-demand features improved predictive signal but required regularization/control through model hyperparameters (e.g., `max_depth`, `min_samples_split`, `C`, `epsilon`) to avoid overfitting.

---

## 5. Variable Types and Encoding Strategy (Question 3)

### 5.1 Selected Variables

- Numerical variable 1: `unit_sales` (target)
- Numerical variable 2: `cluster` (or `class`)
- Categorical variable: `family` (also `state`, `type`)

### 5.2 Encoding Methods for Regression

- **One-hot encoding**: applied to nominal categories (`family`, `state`, `type`) to avoid artificial ordering.
- **Label encoding**: acceptable for identifiers used as grouping keys in engineered features, but not ideal as direct ordinal predictors.
- **Ordinal encoding**: only appropriate when category order has true semantics (not the case for `family` or `state` here).

### 5.3 Statistical Justification

One-hot encoding preserves unbiased linear separability among category levels and avoids injecting false monotonic structure. This reduces misspecification risk in regression and improves compatibility with kernel and tree models after preprocessing.

---

## 6. Support Vector Regression Experiments (Question 4)

### 6.1 Setup

- Kernel: RBF  
- Training split: 80/20  
- Numerical preprocessing: median imputation + standard scaling  
- Categorical preprocessing: mode imputation + one-hot encoding  
- SVR trained on a representative training subset for computational tractability

### 6.2 Hyperparameter Comparison

| C | Epsilon | RMSE | MAE | R² |
|---:|---:|---:|---:|---:|
| 1.0 | 0.1 | 11.7217 | 4.0112 | 0.2489 |
| 10.0 | 0.1 | 10.8812 | 3.9006 | 0.3528 |
| 10.0 | 0.5 | 10.8762 | 3.8944 | 0.3533 |
| 50.0 | 0.1 | 10.5313 | 3.9350 | 0.3937 |

### 6.3 Discussion

Increasing `C` improves fit (lower RMSE, higher R²) by penalizing large residuals more strongly. Adjusting `epsilon` from 0.1 to 0.5 produced only marginal change in this dataset.

### 6.4 SVM Advantages and Limitations

**Advantages**
- Captures nonlinear relationships via kernels  
- Effective in high-dimensional transformed feature spaces  
- Relatively robust to moderate noise with proper margin settings

**Disadvantages**
- Computationally expensive for large-scale datasets  
- Sensitive to hyperparameter tuning (`C`, `epsilon`, kernel settings)  
- Lower interpretability than tree-based rules

---

## 7. Decision Tree Regression Experiments (Question 5)

### 7.1 Setup and Hyperparameters

Core hyperparameters:

- `max_depth`: controls tree complexity and overfitting tendency
- `min_samples_split`: minimum observations to split an internal node

### 7.2 Performance Comparison

| max_depth | min_samples_split | RMSE | MAE | R² |
|---:|---:|---:|---:|---:|
| 8 | 50 | 9.6142 | 4.1544 | 0.4947 |
| 12 | 30 | **9.3373** | **4.1403** | **0.5234** |
| None | 20 | 9.5220 | 4.3769 | 0.5044 |

Best setting: **`max_depth=12`, `min_samples_split=30`**.

### 7.3 Internal Split and Terminal Node Interpretation

From the trained tree (truncated rule extract), one dominant early split is:

- `item_mean_sales <= 14.39`

Interpretation: the model first separates low-demand and high-demand products by historical mean item sales, confirming that product-level baseline demand is a primary determinant.

Example terminal-node interpretation:

- Example observation reached leaf node **785**
- Leaf prediction (mean target at leaf): **3.934**
- True value for that instance: **1.000**

This demonstrates how each terminal node returns an average demand estimate conditioned on the sequence of split rules.

---

## 8. Model Evaluation and Recommendation

### 8.1 Overall Comparison

- Best SVR RMSE: **10.5313** (R² = **0.3937**)
- Best Decision Tree RMSE: **9.3373** (R² = **0.5234**)

Decision Tree achieves substantially better error and explanatory power on this dataset.

### 8.2 Recommendation for Practical Deployment

For practical deployment, **Decision Tree Regression (depth=12, min_samples_split=30)** is recommended because:

1. Higher predictive performance than tested SVR settings  
2. Strong interpretability through explicit split rules  
3. Easier communication of business logic to non-technical stakeholders

SVR remains a useful benchmark but is less favorable here due to weaker predictive results and lower interpretability.

---

## 9. Conclusion

This project successfully implemented a complete machine learning forecasting pipeline, from statistical exploration to predictive modeling and interpretation. The dataset is clean and information-rich, with meaningful temporal and hierarchical demand patterns. Feature engineering (especially demand-history aggregates) was critical for predictive quality. Among evaluated models, Decision Tree Regression provided the best trade-off between accuracy and interpretability, making it the preferred model for final deployment in this retail sales forecasting context.

---

## References

- James, G., Witten, D., Hastie, T., & Tibshirani, R. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
- Géron, A. (2022). *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
