HEAD
# Instamart Customer Intelligence & Support Copilot

## Overview

An end-to-end, reproducible customer intelligence, churn prediction, automated ticket triage, policy RAG assistant, and autonomous refund agent system for an Indian quick-commerce grocery delivery platform (Instamart). Built in strict accordance with the capstone project specifications, verified against deterministic ground-truth benchmarks, and constrained to Indian Rupees (INR) and offline, local computation without external APIs or hosted LLMs.

The Copilot integrates four interconnected subsystems:
1. **Part A (Data Foundation: Clean, Store, Query)**: Ingests and audits dirty raw customer, product, and order exports; applies deterministic cleaning rules to rectify invalid values, multi-status duplicates, and missing timestamps; persists clean relational data into SQLite (`data/instamart.db`); and executes complex analytical SQL queries alongside pandas exploratory data analysis.
2. **Part B (Customer Segments & Lapse Prediction)**: Quantifies customer behavior via RFM analysis, segments users into actionable cohorts using K-Means clustering ($K=4$) and PCA, builds a leakage-safe predictive feature store, and trains, tunes, balances, and persists a production Random Forest churn classifier.
3. **Part C (Support Ticket Triage with Guardrails)**: Implements privacy guardrails (regex PII masking), input token ceilings, exponential backoff retries with jitter for simulated transient 503 errors, output validation, token budgeting, and few-shot routing leveraging customer segments from Part B.
4. **Part D (Policy RAG Assistant & Autonomous Refund Agent)**: Chunks store policy documents, builds a TF-IDF cosine-similarity retrieval engine, evaluates grounded question-answering with fallback handling, conducts a zero-overlap ablation study, and deploys a checkpointed, crash-resilient ReAct refund agent equipped with tool-boundary financial ceilings and ledger idempotency.

---

## Reproduce

The entire repository pipeline can be reproduced sequentially from a clean workspace using the numbered commands below:

```bash
# 1. Create and activate a Python virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
# source .venv/bin/activate

# 2. Upgrade pip and install pinned project dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Generate synthetic raw data, policies, tickets, and scenarios
python generate_data.py

# 4. Part A: Audit dirty raw data exports
python src/data/audit.py

# 5. Part A: Clean customers and orders datasets
python src/data/clean_customers.py
python src/data/clean_orders.py

# 6. Part A: Build SQLite database and run automated reference validation
python src/data/build_database.py
python src/data/validate_part_a.py

# 7. Part A: Run seven SQL analytical queries and pandas EDA cross-checks
python src/analytics/run_sql.py
python src/analytics/eda.py

# 8. Part B: Customer segmentation via RFM, K-Means clustering, and PCA
python src/analytics/customer_segmentation.py

# 9. Part B: Build leakage-safe lapse prediction cohort dataset
python src/ml/build_lapse_dataset.py

# 10. Part B: Preprocess, train, tune, balance, interpret, and persist lapse model
python src/ml/train_lapse_model.py

# 11. Part C: Support ticket triage pipeline with guardrails, retries, and routing
python src/support/triage.py

# 12. Part D: Evaluate Policy RAG Assistant and zero-overlap ablation study
python src/rag/evaluate_rag.py

# 13. Part D: Run Autonomous Refund Agent across scenarios, safety stops, and crash recovery
python src/agent/run_agent.py
```

---

## Part A — Data Foundation: Clean, Store, Query

### 1. Raw Data Audit Profiles & Checkpoints

The three dirty raw data exports were loaded with pandas (`tickets.csv` loaded with `keep_default_na=False`). Below are the shapes, column data types, missing value counts, and missing percentages for every column across all three tables:

#### Table Profiles

**1. `customers.csv` (Shape: 1,200 rows, 5 columns)**
| Column Name | Raw Dtype | Missing Count | Missing % | Data Quality & Anomaly Notes |
| :--- | :---: | :---: | :---: | :--- |
| `customer_id` | `object` (`str`) | 0 | 0.00% | Primary key, C0001–C1200 |
| `city` | `object` (`str`) | 0 | 0.00% | 24 raw casing and whitespace variations across 6 cities |
| `signup_date` | `object` (`str`) | 0 | 0.00% | ISO format strings (YYYY-MM-DD) |
| `age` | `object` (`str`) | 61 | 5.08% | Blank values; plus 6 impossible planted ages (-4, 0, 150, 200, 280, 999) |
| `gender` | `object` (`str`) | 30 | 2.50% | Blank values; plus 10 raw spelling/casing variations |

**2. `products.csv` (Shape: 40 rows, 4 columns)**
| Column Name | Raw Dtype | Missing Count | Missing % | Data Quality & Anomaly Notes |
| :--- | :---: | :---: | :---: | :--- |
| `product_id` | `object` (`str`) | 0 | 0.00% | Primary key, P001–P040 |
| `product_name` | `object` (`str`) | 0 | 0.00% | Clean product description |
| `category` | `object` (`str`) | 0 | 0.00% | 5 categories: Grocery, Dairy, Snacks, Personal Care, Electronics |
| `price_inr` | `int64` | 0 | 0.00% | Unit catalogue price in INR (ranging from INR 20 to INR 1999) |

**3. `orders.csv` (Shape: 13,084 rows, 9 columns)**
| Column Name | Raw Dtype | Missing Count | Missing % | Data Quality & Anomaly Notes |
| :--- | :---: | :---: | :---: | :--- |
| `order_id` | `object` (`str`) | 0 | 0.00% | 12,309 unique IDs; 119 exact duplicates; 656 pending rows |
| `customer_id` | `object` (`str`) | 0 | 0.00% | Foreign key referencing `customers.csv` |
| `product_id` | `object` (`str`) | 0 | 0.00% | Foreign key referencing `products.csv` |
| `quantity` | `int64` | 0 | 0.00% | Units purchased (1 to 3) |
| `order_ts` | `object` (`str`) | 0 | 0.00% | Timestamps spanning 2025-04-01 to 2025-09-30 |
| `status` | `object` (`str`) | 0 | 0.00% | Completed, cancelled, or pending |
| `discount_pct` | `int64` | 0 | 0.00% | Percentage discount applied (0% to 40%) |
| `order_amount` | `object` (`str`) | 0 | 0.00% | 120 unparseable strings (67 `'error'`, 53 `'N/A'`) |
| `delivery_minutes` | `object` (`str`) | 2,541 | 19.42% | Blank values; plus 106 negative durations (< 0) |

#### Reference Audit Checkpoint Comparison
| Category | Specific Audit Metric Description | Computed | Reference | Status |
| :--- | :--- | :---: | :---: | :---: |
| **ORDERS** | Raw `orders.csv` rows | 13,084 | 13,084 | **PASS** |
| **ORDERS** | Exact duplicate rows | 119 | 119 | **PASS** |
| **ORDERS** | Rows with `pending` status | 656 | 656 | **PASS** |
| **ORDERS** | Distinct `order_id` count | 12,309 | 12,309 | **PASS** |
| **ORDERS** | Non-numeric `order_amount` entries (`'N/A'` or `'error'`) | 120 | 120 | **PASS** |
| **ORDERS** | Negative `delivery_minutes` (< 0) | 106 | 106 | **PASS** |
| **ORDERS** | Blank `delivery_minutes` | 2,541 | 2,541 | **PASS** |
| **CUSTOMERS** | Blank `age` values | 61 | 61 | **PASS** |
| **CUSTOMERS** | Age values outside 10–100 | 6 | 6 | **PASS** |
| **CUSTOMERS** | Distinct raw city spellings | 24 | 24 | **PASS** |
| **CUSTOMERS** | Distinct non-blank raw gender spellings | 10 | 10 | **PASS** |
| **CUSTOMERS** | Blank `gender` values | 30 | 30 | **PASS** |

---

### 2. Cleaning Transformations & Row Impact

- **Customers** (`src/data/clean_customers.py`):
  1. Standardized raw city strings by trimming whitespace and converting to title case, collapsing 24 variations into 6 metro cities: Pune (227), Delhi (210), Bengaluru (197), Chennai (191), Hyderabad (191), Mumbai (184).
  2. Standardized gender spellings into three canonical values: male (560), female (610), and unknown (30).
  3. Replaced 6 out-of-bounds ages (<10 or >100) with missing, then imputed all 67 missing ages using city-level median age via `groupby('city')['age'].transform('median')`. Result: 0 missing ages remain; all customer ages range between 18 and 60.
  4. Converted `signup_date` strings to ISO datetimes.
- **Orders** (`src/data/clean_orders.py`):
  1. Converted `order_ts` to datetime.
  2. Dropped 119 exact duplicate rows.
  3. Deduplicated multi-status records for the same `order_id` by retaining only the row with the latest `order_ts`. Exactly 12,309 unique orders remain (11,110 completed, 1,199 cancelled, 0 pending).
  4. Replaced 120 non-numeric amount entries with calculated values using: `round(price_inr * quantity * (1 - discount_pct / 100), 2)`.
  5. Converted 106 negative delivery minutes to missing. Imputed missing delivery times for completed orders using the median delivery time of the product's category. For cancelled orders, delivery duration was left legitimately missing (0 missing among completed; 1,199 missing among cancelled).
- **Completed Spend Grand Total**: Exactly **INR 2,998,668.80**.

---

### 3. Database & SQL Analytical Checkpoints (`data/instamart.db`)

Clean tables were saved to `data/clean/` and written to SQLite database `data/instamart.db`. Running `src/data/validate_part_a.py` asserts all cleaning invariants and prints `ALL CHECKPOINTS PASSED`.

The seven queries in `sql/queries.sql` returned exact reference results:
- **Q1 (WHERE & HAVING)**: Completed orders with `order_amount > 500` grouped by customer with spend `> 8000`: **30 customers**.
- **Q2 (3-Table JOIN)**: Completed revenue by city and category: **30 rows**, grand total **INR 2,998,668.80**, largest cell **Bengaluru / Electronics** at **INR 233,909.45**.
- **Q3 (Scalar Subquery)**: Products priced above catalogue average (**INR 289.33**): **11 products** (P038 1999, P033 1499, P034 1299, P036 799, P037 599, P040 549, P032 399, P030 350, P039 349, P014 340, P035 299).
- **Q4 (LEFT JOIN)**: Customers with 0 completed orders: **25 customers**.
- **Q5 (CTEs)**: Customers above average customer spend (INR 2,552.06): **550 customers**.
- **Q6 (Window Functions)**: Top customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Exactly 8 rows differ between `RANK()` and `ROW_NUMBER()` because ties in customer spend receive identical ranks under `RANK()` while `ROW_NUMBER()` increments sequentially.
- **Q7 (Running Total)**: 183 daily rows; final cumulative revenue is **INR 2,998,668.80**.

---

### 4. Pandas Exploratory Data Analysis

- **Pivot Cross-Check**: `pd.pivot_table(completed, values='order_amount', index='city', columns='category', aggfunc='sum', margins=True)` equals SQL grand total to the penny (**INR 2,998,668.80**).
- **City Aggregations**:
  - Bengaluru: sum INR 523,643.40, mean INR 276.47, count 1,894
  - Chennai: sum INR 463,629.15, mean INR 272.72, count 1,700
  - Delhi: sum INR 502,829.80, mean INR 243.62, count 2,064
  - Hyderabad: sum INR 528,589.90, mean INR 284.95, count 1,855
  - Mumbai: sum INR 479,058.50, mean INR 297.00, count 1,613
  - Pune: sum INR 500,918.05, mean INR 252.48, count 1,984
- **Order Amount Distribution**: Mean INR 269.91, median INR 135.00, skewness 4.61 (right-skewed), Q1 = INR 64.80, Q3 = INR 297.00, IQR = INR 232.20, lower fence = INR -283.50 (0 outliers below), upper fence = INR 645.30 (897 outliers above).
- **Correlation Matrix**: Strongest absolute correlation is between `order_amount` and `discount_pct` at **-0.0951** (followed by `quantity` at **+0.0873**, and `delivery_minutes` at **-0.0031**).

#### Written Interpretation (176 words — within 200 words maximum)
This dataset represents transactional order fulfillment across six Indian metro cities for Swiggy Instamart over six months.

The order amount distribution is highly right-skewed (skewness 4.61), where the mean (INR 269.91) is double the median (INR 135.00). Most orders consist of daily low-cost essentials in Grocery, Dairy, and Snacks (Q1 = INR 64.80, Q3 = INR 297.00).

Using the 1.5xIQR rule (IQR = INR 232.20), the upper fence is INR 645.30, identifying 897 upper outliers (about 8.1% of completed orders) and 0 lower outliers. These high-value outliers are driven by premium electronic devices (e.g., Bluetooth speakers, power banks) and bulk personal care purchases.

Linear relationships among numerical features are weak. The strongest absolute correlation is between order amount and discount percentage (r = -0.0951), followed by order amount and quantity (r = +0.0873). Delivery duration shows virtually zero correlation with order size (r = -0.0031). Crucially, these modest statistical correlations do not establish causation: promotional discounts or delivery speeds do not mechanically drive basket values without considering confounding factors like category-specific promotions and customer purchasing archetypes.

---

## Part B — Customer Segments and Lapse Prediction

### 1. RFM Analysis & K-Means Clustering

- **Cohort**: 1,175 customers with completed orders as of snapshot `2025-10-01 00:00:00`.
- **Metrics**: Recency median 19.0 days (0–180); frequency mean 9.46 (max 33); monetary mean INR 2,552.06 (max INR 14,766.55).

#### Full K-Means Results Grid ($K=2 \dots 8$)
Standardized via `StandardScaler`, fitted with `n_init=10`, `random_state=42`:

| Clusters ($K$) | Inertia | Reference Inertia ($\pm 5\%$) | Silhouette Score | Reference Silhouette ($\pm 0.05$) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 1820.8 | 1820.8 | 0.465 | 0.465 | Coarse segmentation |
| **3** | 1268.2 | 1268.2 | 0.445 | 0.445 | Sub-optimal separation |
| **4** | **743.1** | **743.1** | **0.526** | **0.526** | **Selected (Elbow bend & peak silhouette)** |
| **5** | 617.4 | 617.4 | 0.480 | 0.481 | Silhouette decreases |
| **6** | 503.2 | 503.2 | 0.482 | 0.482 | Plateau |
| **7** | 408.5 | 408.5 | 0.416 | 0.416 | Declining distinctiveness |
| **8** | 366.5 | 366.5 | 0.412 | 0.412 | Over-segmented |

**Why $K=8$ inertia is lower than $K=2$ yet cannot alone pick $K$**:
Mathematically, inertia (within-cluster sum of squared errors) decreases monotonically with increasing $K$ because adding cluster centroids always reduces or preserves Euclidean distance to the nearest center, trivially reaching zero when $K=N$. Minimizing inertia alone overfits the data into fragmented micro-clusters lacking operational utility. At $K=4$, the silhouette score peaks at 0.526 and an elbow bend emerges, yielding four distinct, actionable business segments.

#### Segment Profiles ($K=4$) & Arbitrary Labeling Note
| Segment ID | Segment Name | Customer Count | Mean Recency (days) | Mean Frequency | Mean Monetary (INR) | Marked Premium? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **0** | Steady Standard Shoppers | 436 | 28.9 | 3.8 | INR 1,057.62 | NO |
| **1** | Premium Spenders | 138 | 24.1 | 6.8 | **INR 7,129.70** | **YES** |
| **2** | Loyal Frequent Shoppers | 419 | 8.8 | 19.4 | INR 3,410.02 | NO |
| **3** | Lapsed / At-Risk Occasionals | 182 | 109.0 | 2.2 | INR 685.99 | NO |

> [!NOTE]
> **Cluster IDs are Arbitrary Labels, Not Rankings**: Cluster numbers (0, 1, 2, 3) are nominal identifiers generated during K-Means initialization. They possess no intrinsic ordinal hierarchy or rank ordering; Segment 3 is not worse than Segment 0, and Segment 1 is not ranked first.

- **Recommended Strategic Actions**:
  - Segment 0: Cross-sell bundle promotions across Personal Care and Snacks.
  - Segment 1: VIP concierge routing, priority support desk, and exclusive early access.
  - Segment 2: Subscription loyalty perks, free quick delivery vouchers, and weekly pantry reminders.
  - Segment 3: Win-back reactivation campaigns with high-discount coupons on favorite categories.
- **PCA 2D Projection**: PC1 explained variance = 0.6343, PC2 = 0.2180 (Cumulative = 0.8523). Output: `outputs/customer_segments.csv` (1,175 rows).

---

### 2. Leakage-Safe Lapse Prediction

- **Cutoff Date**: `2025-09-01`. Cohort: 1,156 customers with completed orders prior to cutoff.
- **Lapse Target**: 443 lapsed (38.32%, no completed orders in September 2025), 713 retained (61.68%).
- **Split**: Train 924 rows, Test 232 rows (stratified on $y$, `random_state=42`).
- **Naive Baseline**: Predicting majority class ("not lapsed") yields accuracy 61.64% and lapsed-class F1 of 0.0.

#### Hyperparameter Tuning & Test Set Confusion Matrices
- Tuned Logistic Regression ($C=100$): 5-Fold CV F1 = 0.6506.
- Tuned Random Forest (`max_depth=8`, `min_samples_leaf=10`): 5-Fold CV F1 = 0.6865.

**1. Tuned Logistic Regression Confusion Matrix (Untouched Test Set, $N=232$):**
| Actual Class \ Predicted Class | Predicted Retained (Class 0) | Predicted Lapsed (Class 1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (Class 0)** | **111** (True Negative) | **32** (False Positive) | **143** |
| **Actual Lapsed (Class 1)** | **31** (False Negative) | **58** (True Positive) | **89** |
| **Total Predicted** | **142** | **90** | **232** |

*Test Metrics (Lapsed Class)*: Precision = 0.6444, Recall = 0.6517, F1 = 0.6480, Accuracy = 72.84%.

**2. Tuned Random Forest Confusion Matrix (Untouched Test Set, $N=232$):**
| Actual Class \ Predicted Class | Predicted Retained (Class 0) | Predicted Lapsed (Class 1) | Total Actual |
| :--- | :---: | :---: | :---: |
| **Actual Retained (Class 0)** | **111** (True Negative) | **32** (False Positive) | **143** |
| **Actual Lapsed (Class 1)** | **32** (False Negative) | **57** (True Positive) | **89** |
| **Total Predicted** | **143** | **89** | **232** |

*Test Metrics (Lapsed Class)*: Precision = 0.6404, Recall = 0.6404, F1 = 0.6404, Accuracy = 72.41% (exceeds $\ge 0.55$ requirement).

#### Model Selection & Asymmetric Cost Justification
Random Forest (`max_depth=8`, `min_samples_leaf=10`) is selected as the production model. In grocery quick-commerce, missing an at-risk customer (a false negative) results in permanent churn and lost customer lifetime value, whereas acquiring a replacement customer is significantly more costly than sending retention vouchers. Conversely, incorrectly targeting a loyal customer (a false positive) incurs only minimal coupon costs without harming the relationship. Random Forest delivers superior 5-fold cross-validation performance (0.6865 vs 0.6506 for Logistic Regression) by capturing non-linear interactions across recency, basket size, and delivery duration, while maintaining balanced test precision and recall (F1 = 0.6404 $\ge 0.55$ requirement).

#### Class-Balancing Experiment (Training Set Oversampling)
Oversampling the training partition to 570/570 balanced classes surged Logistic Regression test recall from **65.17% to 80.90%** (Precision 0.6000, F1 0.6890, Test Accuracy 71.98%).
- Balanced Logistic Regression Confusion Matrix: `[TN=95, FP=48 / FN=17, TP=72]`.
- *Takeaway*: Recall changed the most (+15.73%). The balanced model captures over 80% of all genuine churn, allowing the retention team to intercept nearly the entire lapsing volume.

#### Feature Interpretation & Persistence
- **Top 5 Random Forest Features**: `monetary` (0.2778), `frequency` (0.2768), `recency` (0.1112), `avg_delivery_minutes` (0.0616), `avg_discount_pct` (0.0532).
- **Logistic Regression Coefficients**: `frequency` (-1.6982, habitual purchase cadence decreases lapse log-odds), `recency` (+0.0396, purchase dormancy increases lapse log-odds), `monetary` (+0.2846, holding frequency constant, high basket spend correlates with occasional electronics purchases).
- **Model Persistence**: Saved to `outputs/lapse_model.joblib`. Reloading via `joblib.load()` reproduces test predictions with 100.0% equality.
- **Retention Ranking**: Top 10 test Premium customers ranked in `outputs/retention_list.csv` (1 actual lapse).

---

### 3. Part B Data Leakage Safeguards

To guarantee sound predictive validity and eliminate data snooping leakage, three explicit safeguards are enforced in code:
1. **Pre-Cutoff Temporal Boundary**: All predictive features (`recency`, `frequency`, `monetary`, `avg_discount_pct`, `cancellation_rate`, `avg_delivery_minutes`, `favourite_category`, `tenure_days`) were constructed strictly and exclusively from order records dated prior to `2025-09-01 00:00:00`. Zero orders dated on or after 2025-09-01 entered feature extraction; September 2025 orders were used solely to evaluate ground-truth retention and construct the binary target label `lapsed`.
2. **Exclusion of Segment Labels**: The cluster assignments and RFM segment labels from Task B1 (`segment_id`, `segment_name`, `is_premium`) were strictly excluded from feature matrix $X$. Because the Task B1 RFM snapshot (`2025-10-01`) incorporated September orders, including cluster labels as features would leak future activity into pre-cutoff predictions.
3. **Strict Train-Only Fitting (Preprocessing & Resampling)**: Preprocessing pipelines (`StandardScaler`, `OneHotEncoder(drop='first')`) and class-balance oversampling were fitted exclusively on the 924 training rows. The 232 test partition rows were never seen during fitting or resampling. As programmatic proof, the scaler's learned mean of `frequency` on the training set is **8.1147**, which differs from the full-cohort frequency mean of **8.0285** due to the holdout partition. Code asserts 0 missing values remain after preprocessing.

---

## Part C — Support Ticket Triage with Guardrails

### 1. Guardrails & Privacy Redactions (`data/raw/tickets.csv`)

- **Input Guardrails**: Evaluated over all 164 raw tickets:
  - `skipped_empty`: 4 tickets (blank or whitespace-only; never sent to model).
  - `flagged_long`: 4 tickets (exceeded 400 token ceiling, estimated at 734 tokens; never sent to model).
  - Eligible tickets processed: 156 tickets.
- **PII Redactions**: Applied regex patterns sequentially before prompt construction:
  - Email (`[\w.+-]+@[\w-]+\.[\w.]+`): 55 substitutions (`[EMAIL]`)
  - Phone (`\b[6-9]\d{9}\b`): 91 substitutions (`[PHONE]`)
  - Order ID (`ORD-\d{6}`): 132 substitutions (`[ORDER_ID]`)
  - **Total Substitutions**: Exactly **278**.
  - Re-applying regex patterns to redacted texts yields 0 unredacted PII matches. User message sent to mock LLM receives strictly the redacted ticket text.

---

### 2. Resilience, Output Validation & Comparison

- **Transient Error Handling**: Encountered 19 simulated `TransientLLMError` (503 server busy) faults; all 19 were successfully recovered via exponential backoff with jitter (`BASE_DELAY = 0.05s`, jitter bounds $[0.045\text{s}, 0.0575\text{s}]$). Zero tickets suffered permanent failure. Logged to `outputs/triage_retry_log.txt`.
- **Output Validation Unit Tests**: Verified against 7 unit tests covering valid JSON allowed, valid JSON unsupported, JSON array, JSON string, invalid JSON single keyword, invalid JSON multi-keyword, and invalid JSON zero keyword. Keyword fallback is bypassed when JSON parsing succeeds.
- **Output Status Breakdown (156 Eligible Tickets)**:
  - `classified_json`: 135 tickets (86.5%)
  - `classified_keyword`: 12 tickets (7.7%)
  - `flagged_invalid_output`: 9 tickets (5.8%) -> routed to human review.
  - Total Classified: 147 tickets.

#### Zero-Shot Confusion Matrix (147 Classified Tickets)
| True Category \ Predicted Category | billing | delivery | technical | fraud | Total Actual |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **billing** | **38** | 0 | 0 | 0 | **38** |
| **delivery** | 0 | **36** | 0 | 0 | **36** |
| **technical** | 10 | 0 | **29** | 0 | **39** |
| **fraud** | 11 | 0 | 0 | **23** | **34** |
| **Total Predicted** | **59** | **36** | **29** | **23** | **147** |

#### Prompt Economic Trade-Off Comparison
| Evaluation Metric | Zero-Shot Prompt | Few-Shot Prompt | Reference Target |
| :--- | :---: | :---: | :---: |
| **Classified Tickets** | 147 | 147 | 147 tickets |
| **Correct Classifications** | 126 | 147 | 126 vs 147 |
| **Classification Accuracy** | **85.71%** | **100.00%** | 85.7% vs 100.0% |
| **Total Tokens Consumed** | 12,364 | 32,332 | — |
| **Tokens per Classified Ticket** | **84.1** | **219.9** | +135.8 tokens/ticket |

**Economic Trade-Off Justification**:
While few-shot prompting consumes 219.9 tokens per ticket compared to 84.1 for zero-shot (a ~2.6x increase), it achieves perfect 100% accuracy (147/147) by completely eliminating severe misclassifications where fraud tickets (11) and technical outages (10) were misrouted to billing. In quick-commerce operations, misrouting fraud tickets to standard billing queues delays account security lockouts and causes catastrophic financial loss. The marginal cost of additional context tokens is heavily outweighed by the elimination of critical safety misroutings.

---

### 3. Token Budget Demonstration

- Configured demonstration with `TOKEN_BUDGET = 6000` tokens using the zero-shot pipeline.
- The pipeline halted after 84 model calls (consuming 5,957 tokens).
- The remaining 80 tickets received status `budget_exhausted` and were safely routed to human review, saved in `outputs/triage_budget_demo.csv`.

---

### 4. Rule-Validated Tickets vs. Human Review Routing

The pipeline rigorously separates automated rule-validated processing from safe fallbacks across all 164 tickets in `outputs/triage_results.csv`:

1. **Automatically Rule-Validated Tickets (147 tickets / 89.6%)**:
   - `classified_json` (**135 tickets**): Response parsed as valid JSON, conformed to schema `{"category": "...", "reason": "..."}`, and returned an allowed category (`billing`, `delivery`, `technical`, `fraud`).
   - `classified_keyword` (**12 tickets**): Response was non-JSON (chatty stand-in message `The customer appears to have a <category>-related problem.`), but regex word extraction `re.findall(r"[a-z]+", text.lower())` identified exactly one distinct allowed category.
   - *Routing Destinations*:
     - `support_queue` (**102 tickets**, NORMAL priority): Standard non-fraud, non-premium tickets.
     - `security_team` (**34 tickets**, HIGH priority): Fraud / unauthorized transaction tickets (target SLA: within 2 hours).
     - `premium_desk` (**11 tickets**, HIGH priority): Tickets from verified Part B Premium customers (target SLA: within 1 hour).
2. **Tickets Routed to Human Review (17 tickets / 10.4%, NORMAL priority)**:
   - `skipped_empty` (**4 tickets**): Empty or whitespace-only bodies; model was never called.
   - `flagged_long` (**4 tickets**): Text exceeded 400 token ceiling (734 estimated tokens); model was never called to safeguard token budget.
   - `flagged_invalid_output` (**9 tickets**): Output parsing rejected due to invalid schema, unsupported out-of-domain categories (e.g. `{"category": "shipping"}`), or ambiguous multi-category responses. The pipeline strictly avoids guessing and routes directly to human leads.
- **Priority Breakdown Across All 164 Tickets**: NORMAL = **119 tickets**, HIGH = **45 tickets**.

---

## Part D — Policy RAG Assistant and Refund Agent

### 1. Policy RAG Assistant (Tasks 1–4)

- **Document Chunking** (Word size = 40, Overlap = 10, Stride = 30):
  - Total chunks generated across 5 documents: **15 chunks**.
  - Document breakdown: `refund_policy.txt` (4), `delivery_policy.txt` (3), `escalation_policy.txt` (3), `payment_policy.txt` (3), `privacy_policy.txt` (2).
- **TF-IDF & Cosine Similarity Retrieval**:
  - Reference query `"refund policy late delivery"` retrieves top 3 chunks originating exclusively from `refund_policy.txt` (similarity scores: 0.2719, 0.2514, 0.2454).
  - *Theoretical Definitions*:
    1. **Term Frequency (TF)** measures local term prominence within an individual chunk, normalized for chunk length.
    2. **Inverse Document Frequency (IDF)** measures the global rarity and discriminatory informativeness of a term across the corpus.
    3. **Limitation**: Pure lexical surface matching fails on semantic paraphrasing (e.g. "reimbursement" vs. "refund").
- **Question-Answering Benchmark** (`data/raw/policy_eval.csv`, 14 questions):
  - Expected Document Retrieved: **12 / 12 (100.0%)** for answerable questions.
  - Expected Fact Accuracy: **12 / 12 (100.0%)** exact fact string match in generated answer.
  - Graceful Fallback Rate: **2 / 2 (100.0%)** unanswerable questions returned *"I could not find that information in the policy documents."*
- **Zero-Overlap Ablation Study** (Chunk size = 40, Overlap = 0):
  - Yields 13 total chunks. Fact accuracy drops from 12/12 down to **10/12**.
  - *Root-cause failure*: Question 1 (*"What delivery time makes a completed order eligible for a full refund?"*) fails because the sentence containing *"more than 45 minutes"* was split across the 40-word boundary, severing the condition clause from the predicate and degrading retrieval.

---

### 2. Five Domain Tools (`src/agent/refund_agent.py`)

1. `get_order(order_id)`: Queries `data/instamart.db` for order metadata. Returns `{"order_id": str, "status": str, "order_amount": float, "delivery_minutes": int}` or structured error `{"error": "order not found"}` if the record does not exist.
2. `search_policy(query)`: Reuses the Part D TF-IDF vectorizer and cosine similarity index over 40-word chunks (overlap 10) to retrieve top 3 relevant passages: `{"results": [{"source": str, "score": float, "text": str}, ...]}`.
3. `calculate_refund(order_amount, delivery_minutes)`: Validates input non-negativity and finiteness (rejects NaN, +/-Inf, non-numeric, or missing values safely). Returns `{"refund_inr": 0.0, "eligible": False, "requires_escalation": False}` when delivery time is $\le 45$ minutes or inputs are invalid. Returns full amount `{"refund_inr": amount, "eligible": True, "requires_escalation": bool}` when delivery time $> 45$ minutes, explicitly flagging claims above INR 1000 as requiring human escalation (`requires_escalation = True`). Pure computation tool; never executes financial disbursements or modifies the ledger.
4. `issue_refund(order_id, refund_inr)`: Enforces tool-boundary guards before modifying state: independently queries `data/instamart.db` to reject unknown orders, non-completed statuses, deliveries $\le 45$ minutes, claims $> \text{INR } 1000.00$, and amounts not matching eligible order spend within INR 0.01. On failure, returns structured rejection `{"error": "<reason>", "refund_inr": 0.0, "status": "rejected"}` without writing to the ledger. On success, executes an idempotency scan to prevent duplicate disbursement writes for the same order, returning `{"refund_inr": amount, "status": "already_issued", "idempotent": True}` on replay, or commits a new JSON line to `outputs/refund_ledger.jsonl` and returns `{"refund_inr": amount, "status": "issued"}`.
5. `escalate_to_human(order_id, refund_inr)`: Enforces independent database validation: verifies order exists, status is `completed`, delivery duration $> 45$ minutes, and amount matches eligible spend within INR 0.01. Strictly rejects claims $\le \text{INR } 1000.00$ (`status: "rejected"`). Executes idempotency check against prior escalation records. If approved, appends an escalation record to `outputs/refund_ledger.jsonl` and returns `{"refund_inr": amount, "status": "escalated"}`.

---

### 3. Seven Benchmark Scenario Results (`outputs/agent_runs.jsonl`)

| Scenario ID | Expected Outcome | Actual Status | Refund (INR) | Model Calls | Tokens Used | Validation Result |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **S1** | REFUND | `REFUND` | 128.25 | 5 | 910 | PASS |
| **S2** | REFUND | `REFUND` | 135.00 | 5 | 910 | PASS |
| **S3** | ESCALATE | `ESCALATE` | 1999.00 | 5 | 897 | PASS |
| **S4** | DENIED | `DENIED` | 0.00 | 4 | 610 | PASS |
| **S5** | DENIED | `DENIED` | 0.00 | 3 | 337 | PASS |
| **S6** | NOT_FOUND | `NOT_FOUND` | 0.00 | 2 | 72 | PASS |
| **S7** | FALLBACK | `FALLBACK_MAX_ITERATIONS` | 0.00 | 6 | 3000 | PASS |

*Notes*:
- S1 took exactly **5 model calls**.
- S7 completed all 6 model calls under default `token_limit=6000` (consuming 3,000 tokens, well within the 2,700–5,500 reference range) before halting cleanly via `FALLBACK_MAX_ITERATIONS`.
- All refund amounts match `expected_refund_inr` within INR 0.01.

---

### 4. Safe Stop Demonstrations (Task 7)

- **Token Limit Demonstration (S1, `token_limit=200`)**: Ended with status **`FALLBACK_TOKEN_LIMIT`**. Cumulative tokens reached 354 after model call 3, halting execution immediately before executing the requested tool (`calculate_refund`). Steps completed: 2 (`get_order`, `search_policy`).
- **Wall-Clock Timeout Demonstration (S1, `time_limit_seconds=0`)**: Ended with status **`FALLBACK_TIMEOUT`**. Evaluated at the top of iteration 1 ($0.00\text{s} \ge 0.0\text{s}$), halting before call 1 (0 model calls, 0 tokens, 0 steps).
- **Maximum Iterations Demonstration (S7, `max_iterations=6`)**: Ended with status **`FALLBACK_MAX_ITERATIONS`** after 6 model calls without final resolution, triggering handoff to a human support lead.

---

### 5. Checkpointing & Crash Recovery Sequence (Task 8)

State checkpointing writes `state/<run_id>.json` after every tool observation is appended. In the simulated crash demonstration on scenario S1 (`crash_run_id = "s1_crash_recovery_demo"`), a crash was raised immediately after the second tool was executed. The execution resumed seamlessly via `resume_agent("s1_crash_recovery_demo")`.

The sequence of tools executed across the pre-crash run and resumed run was:
1. `get_order` *(Pre-crash)*
2. `search_policy` *(Pre-crash)*
3. `calculate_refund` *(Post-resume)*
4. `issue_refund` *(Post-resume)*

**Result**: Exactly **4 tool executions** with **zero repetitions**, culminating in final status **`REFUND`** and refund issued of **INR 128.25**.

---

### 6. ReAct vs. Plan-and-Execute Analysis (Task 9)

**Measured ReAct Performance**:
In our benchmark runs, the ReAct agent completed workflows dynamically with measured model calls and token footprints: full refunds took 5 calls (S1: 910 tokens, S2: 910 tokens), escalation took 5 calls (S3: 897 tokens), on-time denials took 4 calls (S4: 610 tokens), non-completed order denials took 3 calls (S5: 337 tokens), and missing orders stopped after 2 calls (S6: 72 tokens). Cumulative tokens grew quadratically with iterations because prior conversation turns and tool observations were re-sent with each step.

**Plan-and-Execute (Theoretical Estimates)**:
A plan-and-execute agent separates planning from execution. An initial planner call emits a static sequence of tool invocations (get_order -> search_policy -> calculate_refund -> issue_refund), executed deterministically by code without intervening model calls. For a standard happy path, this requires an estimated 1 planner call (~120 tokens) and 1 optional response synthesis call (~150 tokens), yielding an estimated ~270 tokens total—saving roughly 70% in prompt tokens compared to ReAct.

**Handling Dynamic & Unexpected Observations**:
However, plan-and-execute struggles when intermediate facts contradict the initial plan. In scenario S6 (ORD-999999 not found) or S5 (order cancelled), an upfront 4-step plan is invalid from step 1. Plan-and-execute must abort and invoke an extra replanning model call, negating its token advantage. Conversely, ReAct responds immediately to step-level observations, halting early without wasted execution.

**Workflow Choice**:
For financial refund processing, ReAct is strongly preferable. The strict prerequisite dependencies (verification, policy lookup, amount checks, escalation thresholding > INR 1000) demand stepwise evaluation and checkpointed auditability before committing money.

---

### 7. Tool-Boundary Security & Idempotency Safeguards

1. **Independent Verification**: `issue_refund` and `escalate_to_human` independently query `data/instamart.db` to verify order existence, completed status, delivery duration (> 45 min), and exact order amount match.
2. **Ceiling Enforcement**: Automatic issuance (`issue_refund`) strictly rejects any claim exceeding INR 1000.00; `escalate_to_human` enforces that claims $\le$ INR 1000 cannot be escalated.
3. **Idempotency Protection**: Ledger writes check existing transactions by `order_id` and tool type before appending. Any replay or crash-recovery re-invocation returns the verified status with `idempotent=True` without appending duplicate financial ledger lines.

---

## Limitations and Next Steps

1. **Synthetic Data & Deterministic Mock LLM**: All workflows run against deterministic synthetic datasets generated via `generate_data.py` and an offline regex/stem-based mock LLM (`mock_llm.py`). Real-world LLM deployments must handle non-deterministic outputs, API latency spikes, prompt injections, and complex lexical ambiguity.
2. **Post-Hoc Whole-Window Segment Labeling**: In Part B, Task B3, the `is_premium` flag assigned to customers originated from the full October 1, 2025 RFM segmentation, which overlaps with the September 2025 lapse labeling window. Consequently, the top-10 Premium retention list (`outputs/retention_list.csv`) serves an illustrative business campaign purpose rather than a strictly leakage-free out-of-time evaluation.
3. **Lexical TF-IDF vs. Dense Embeddings**: The policy assistant uses sparse TF-IDF with cosine similarity over 40-word chunks. While computationally lightweight and deterministic, it is incapable of semantic synonymy or multilingual translation. A production upgrade should integrate dense neural embeddings (e.g., `text-embedding-3-small` or local BERT/SentenceTransformer models) and hybrid dense-sparse vector search.
4. **Financial Ledger Idempotency & Concurrency**: The financial refund ledger (`outputs/refund_ledger.jsonl`) utilizes a file-based check-before-append idempotency mechanism designed specifically for this single-process offline demonstration. In concurrent production environments, file-based scanning is subject to race conditions and TOCTOU (time-of-check to time-of-use) vulnerabilities; production deployments require ACID transactional database constraints (e.g., unique constraints on `order_id` in a relational transaction table), row-level database locking, or distributed consensus locks (e.g., Redis Redlock).
5. **State Persistence**: State checkpoints are stored as flat JSON files in `state/`. In high-throughput distributed production environments, checkpoints should be committed to a persistent transactional store (such as PostgreSQL or Redis) with distributed locking.
=======
# Instamart-copilot
a72a33930e2bc051694d7e2020259d2533815ba4
