# 🧠 Feature Engineering: Complete Notes (Beginner → Advanced)
### Job-Ready for ML | DL | Gen AI | Agentic AI

> **What is Feature Engineering?**
> Feature Engineering is the process of using **domain knowledge** to transform raw data into features (inputs) that make machine learning algorithms work better. It's the most impactful skill in any ML/AI pipeline.

---

## 📚 Table of Contents

1. [Why Feature Engineering Matters](#1-why-feature-engineering-matters)
2. [Types of Data](#2-types-of-data)
3. [Handling Missing Values](#3-handling-missing-values)
4. [Encoding Categorical Variables](#4-encoding-categorical-variables)
5. [Feature Scaling & Normalization](#5-feature-scaling--normalization)
6. [Outlier Detection & Treatment](#6-outlier-detection--treatment)
7. [Feature Transformation](#7-feature-transformation)
8. [Feature Creation & Interaction Features](#8-feature-creation--interaction-features)
9. [Date & Time Features](#9-date--time-features)
10. [Text Feature Engineering](#10-text-feature-engineering)
11. [Image Feature Engineering](#11-image-feature-engineering)
12. [Feature Selection](#12-feature-selection)
13. [Dimensionality Reduction](#13-dimensionality-reduction)
14. [Handling Imbalanced Datasets](#14-handling-imbalanced-datasets)
15. [Feature Engineering for Deep Learning](#15-feature-engineering-for-deep-learning)
16. [Feature Engineering for NLP & Gen AI](#16-feature-engineering-for-nlp--gen-ai)
17. [Feature Engineering for Agentic AI](#17-feature-engineering-for-agentic-ai)
18. [Feature Stores & MLOps](#18-feature-stores--mlops)
19. [Automated Feature Engineering (AutoFE)](#19-automated-feature-engineering-autofe)
20. [End-to-End Pipeline Example](#20-end-to-end-pipeline-example)
21. [Interview Questions & Cheatsheet](#21-interview-questions--cheatsheet)

---

## 1. Why Feature Engineering Matters

```
Raw Data → Feature Engineering → Model → Predictions
```

> **"Better features = Better models."** Even a simple model with great features beats a complex model with bad features.

### Key Reasons:
- Algorithms learn from numbers — FE translates real-world info into numbers
- Reduces model complexity — fewer, better features = faster training
- Improves accuracy dramatically
- Makes model interpretable

```python
# Simple example: Age is raw. "Is Senior Citizen" is a feature.
import pandas as pd

df = pd.DataFrame({'age': [15, 25, 45, 65, 70]})

# Raw feature
print(df['age'])

# Engineered feature — binary flag
df['is_senior'] = (df['age'] >= 60).astype(int)
print(df)
```

---

## 2. Types of Data

### 2.1 Numerical Data
- **Continuous**: Height, Weight, Temperature (any value in a range)
- **Discrete**: Number of children, count of items (whole numbers)

### 2.2 Categorical Data
- **Nominal**: No order — Color (Red, Blue, Green), Gender
- **Ordinal**: Has order — Education (High School < Bachelor's < Master's < PhD)

### 2.3 Text Data
- Reviews, tweets, articles, product descriptions

### 2.4 Date/Time Data
- Timestamps, dates, seasons

### 2.5 Image/Audio Data
- Pixel values, spectrograms

```python
import pandas as pd
import numpy as np

# Example dataset with all types
df = pd.DataFrame({
    'age': [25, 35, 45],                        # Numerical continuous
    'num_purchases': [1, 3, 5],                  # Numerical discrete
    'city': ['Delhi', 'Mumbai', 'Chennai'],       # Categorical nominal
    'education': ['High School', 'Bachelor', 'Master'],  # Categorical ordinal
    'signup_date': pd.to_datetime(['2022-01-01', '2023-05-10', '2024-03-20']),  # DateTime
    'review': ['Good product', 'Bad quality', 'Average']  # Text
})

print(df.dtypes)
print(df.head())
```

---

## 3. Handling Missing Values

> Missing data is the #1 real-world problem. Handle it wrong → model fails.

### 3.1 Detect Missing Values

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({
    'age': [25, np.nan, 35, np.nan, 45],
    'salary': [50000, 60000, np.nan, 80000, 90000],
    'city': ['Delhi', 'Mumbai', np.nan, 'Chennai', 'Pune']
})

# Count missing values
print(df.isnull().sum())

# Percentage missing
print((df.isnull().sum() / len(df)) * 100)

# Heatmap of missing values
import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False)
plt.show()
```

### 3.2 Drop Missing Values

```python
# Drop rows with ANY missing value
df_dropped = df.dropna()

# Drop rows where ALL values are missing
df_dropped_all = df.dropna(how='all')

# Drop columns with more than 40% missing
threshold = 0.4
df_cleaned = df.dropna(thresh=int((1 - threshold) * len(df)), axis=1)

print(df_cleaned)
```

### 3.3 Mean / Median / Mode Imputation

```python
from sklearn.impute import SimpleImputer
import numpy as np

# Mean imputation — for normal distributed numerical data
mean_imputer = SimpleImputer(strategy='mean')
df[['age']] = mean_imputer.fit_transform(df[['age']])

# Median imputation — for skewed numerical data (better than mean)
median_imputer = SimpleImputer(strategy='median')
df[['salary']] = median_imputer.fit_transform(df[['salary']])

# Mode imputation — for categorical data
mode_imputer = SimpleImputer(strategy='most_frequent')
df[['city']] = mode_imputer.fit_transform(df[['city']])

print(df)
```

### 3.4 Constant / Custom Value Imputation

```python
# Fill with constant
constant_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
df[['city']] = constant_imputer.fit_transform(df[['city']])

# Or use pandas directly
df['age'].fillna(0, inplace=True)         # fill with 0
df['city'].fillna('Unknown', inplace=True) # fill with custom string
```

### 3.5 KNN Imputation (Smart — uses neighbors)

```python
from sklearn.impute import KNNImputer

df_num = pd.DataFrame({
    'age': [25, np.nan, 35, np.nan, 45],
    'salary': [50000, 60000, np.nan, 80000, 90000],
    'experience': [2, 5, 8, np.nan, 15]
})

# KNN Imputer — fills based on K nearest neighbors
knn_imputer = KNNImputer(n_neighbors=2)
df_imputed = pd.DataFrame(knn_imputer.fit_transform(df_num), columns=df_num.columns)

print(df_imputed)
```

### 3.6 Iterative Imputation (Best for complex data)

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Uses each feature to predict the others iteratively
iterative_imputer = IterativeImputer(max_iter=10, random_state=42)
df_imputed = pd.DataFrame(
    iterative_imputer.fit_transform(df_num),
    columns=df_num.columns
)
print(df_imputed)
```

### 3.7 Add "Missing Indicator" Column (Pro Trick)

```python
# Tell the model THAT a value was missing — often very informative!
df['age_was_missing'] = df['age'].isnull().astype(int)
df['age'].fillna(df['age'].median(), inplace=True)

print(df[['age', 'age_was_missing']])
```

### 3.8 When to Use What

| Situation | Strategy |
|-----------|----------|
| < 5% missing, numerical | Mean / Median |
| Skewed distribution | Median |
| Categorical | Mode or 'Unknown' |
| < 30% missing, complex | KNN / Iterative |
| > 50% missing in column | Drop column |
| Missing is informative | Add indicator |

---

## 4. Encoding Categorical Variables

> ML models only understand numbers. Encoding = turning categories → numbers.

### 4.1 Label Encoding (Ordinal — has order)

```python
from sklearn.preprocessing import LabelEncoder

df = pd.DataFrame({'education': ['High School', 'Bachelor', 'Master', 'PhD']})

# Manually map for ordinal (preserves order)
order_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_encoded'] = df['education'].map(order_map)

print(df)
# ✅ Use ONLY when there's a natural order
```

### 4.2 One-Hot Encoding (Nominal — no order)

```python
df = pd.DataFrame({'city': ['Delhi', 'Mumbai', 'Chennai', 'Mumbai', 'Delhi']})

# pandas get_dummies
df_encoded = pd.get_dummies(df, columns=['city'], drop_first=True)
print(df_encoded)

# Sklearn version
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, drop='first')
encoded = ohe.fit_transform(df[['city']])
print(encoded)

# ⚠️ Problem: Too many unique categories → too many columns (high cardinality)
```

### 4.3 Binary Encoding (For high cardinality)

```python
# pip install category_encoders
import category_encoders as ce

df = pd.DataFrame({'city': ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad']})

encoder = ce.BinaryEncoder(cols=['city'])
df_encoded = encoder.fit_transform(df)
print(df_encoded)
# Creates log2(N) columns instead of N columns
```

### 4.4 Target / Mean Encoding (Powerful for high cardinality)

```python
import category_encoders as ce

df = pd.DataFrame({
    'city': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai'],
    'price': [100, 200, 150, 300, 250]
})

# Replaces category with mean of target
encoder = ce.TargetEncoder(cols=['city'])
df['city_encoded'] = encoder.fit_transform(df['city'], df['price'])
print(df)

# ⚠️ Risk: Data leakage — always fit on train, transform on test separately!
```

### 4.5 Frequency / Count Encoding

```python
df = pd.DataFrame({'city': ['Delhi', 'Mumbai', 'Delhi', 'Chennai', 'Mumbai', 'Delhi']})

# Replace category with how often it appears
freq_map = df['city'].value_counts()
df['city_freq'] = df['city'].map(freq_map)
print(df)
```

### 4.6 Ordinal Encoding (Sklearn)

```python
from sklearn.preprocessing import OrdinalEncoder

df = pd.DataFrame({'size': ['S', 'M', 'L', 'XL', 'M', 'S']})

encoder = OrdinalEncoder(categories=[['S', 'M', 'L', 'XL']])
df['size_encoded'] = encoder.fit_transform(df[['size']])
print(df)
```

### 4.7 Hash Encoding (For very high cardinality)

```python
import category_encoders as ce

df = pd.DataFrame({'product_id': ['P001', 'P002', 'P003', 'P999', 'P500']})

encoder = ce.HashingEncoder(cols=['product_id'], n_components=4)
df_encoded = encoder.fit_transform(df)
print(df_encoded)
# Fixed output size regardless of unique values
```

### 4.8 When to Use What

| Encoding | When to Use |
|----------|-------------|
| Label Encoding | Ordinal categories (Low/Med/High) |
| One-Hot | Low cardinality nominal (<10 unique) |
| Binary | Medium cardinality (10–100 unique) |
| Target Encoding | High cardinality + tree models |
| Frequency | High cardinality, interpretable |
| Hash | Very high cardinality (1000s) |

---

## 5. Feature Scaling & Normalization

> Distance-based and gradient-based models (KNN, SVM, Neural Nets, Logistic Regression) are sensitive to feature scales.

### 5.1 Min-Max Scaling (Normalization) → [0, 1]

```python
from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[1000], [2000], [3000], [50000], [100000]])

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)
print(scaled)

# Formula: (x - min) / (max - min)
# ⚠️ Sensitive to outliers — outliers stretch the scale
```

### 5.2 Standardization (Z-score) → mean=0, std=1

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(data)
print(scaled)

# Formula: (x - mean) / std
# ✅ Handles outliers better than MinMax
# ✅ Best for: Linear Regression, Logistic Regression, SVM, PCA
```

### 5.3 Robust Scaler (Best for outlier-heavy data)

```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()
scaled = scaler.fit_transform(data)
print(scaled)

# Formula: (x - median) / IQR
# ✅ Uses median & IQR — not affected by extreme outliers
```

### 5.4 Max Absolute Scaler → [-1, 1]

```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler()
scaled = scaler.fit_transform(data)
print(scaled)

# Divides by maximum absolute value
# ✅ Good for sparse data (doesn't shift/center)
```

### 5.5 L1 / L2 Normalization (Row-wise)

```python
from sklearn.preprocessing import Normalizer

data_2d = np.array([[1, 2, 3], [4, 5, 6]])

# L2 norm — each ROW has unit length
normalizer = Normalizer(norm='l2')
print(normalizer.fit_transform(data_2d))

# L1 norm — row sums to 1
normalizer_l1 = Normalizer(norm='l1')
print(normalizer_l1.fit_transform(data_2d))
# ✅ Good for: text, NLP, cosine similarity
```

### 5.6 When to Scale

| Model | Needs Scaling? |
|-------|---------------|
| Linear/Logistic Regression | ✅ Yes |
| SVM | ✅ Yes |
| KNN | ✅ Yes |
| Neural Networks | ✅ Yes |
| Decision Tree / Random Forest | ❌ No |
| XGBoost / LightGBM | ❌ No (but helps sometimes) |
| Gradient Boosting | ❌ No |

---

## 6. Outlier Detection & Treatment

> Outliers = extreme values that don't fit the pattern. They can destroy models.

### 6.1 Detect with IQR (Interquartile Range)

```python
import pandas as pd
import numpy as np

df = pd.DataFrame({'salary': [30000, 35000, 32000, 28000, 500000, 33000, 31000]})

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print(f"Outliers:\n{df[(df['salary'] < lower) | (df['salary'] > upper)]}")
```

### 6.2 Detect with Z-Score

```python
from scipy import stats

z_scores = np.abs(stats.zscore(df['salary']))
outliers = df[z_scores > 3]   # Z > 3 = outlier
print(outliers)
```

### 6.3 Detect Visually

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Box plot
sns.boxplot(x=df['salary'])
plt.show()

# Histogram
df['salary'].hist(bins=20)
plt.show()
```

### 6.4 Isolation Forest (For high-dimensional data)

```python
from sklearn.ensemble import IsolationForest

data = pd.DataFrame({'salary': [30000, 35000, 32000, 500000, 33000],
                     'age': [25, 30, 28, 75, 27]})

model = IsolationForest(contamination=0.1, random_state=42)
data['outlier'] = model.fit_predict(data)

# -1 = outlier, 1 = normal
print(data)
```

### 6.5 Local Outlier Factor (LOF)

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(n_neighbors=2)
data['outlier_lof'] = lof.fit_predict(data[['salary', 'age']])
print(data)
```

### 6.6 Treatment Strategies

```python
df = pd.DataFrame({'salary': [30000, 35000, 32000, 28000, 500000, 33000, 31000]})

# Strategy 1: Remove outliers
Q1, Q3 = df['salary'].quantile([0.25, 0.75])
IQR = Q3 - Q1
df_no_outlier = df[(df['salary'] >= Q1 - 1.5*IQR) & (df['salary'] <= Q3 + 1.5*IQR)]

# Strategy 2: Cap / Winsorize (clip to boundary)
df['salary_capped'] = df['salary'].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)

# Strategy 3: Log transform (reduces outlier effect)
df['salary_log'] = np.log1p(df['salary'])

# Strategy 4: Replace with median
median_val = df['salary'].median()
df['salary_median_replaced'] = df['salary'].apply(
    lambda x: median_val if x > Q3 + 1.5*IQR else x
)

print(df)
```

---

## 7. Feature Transformation

> Transformations change the distribution/shape of features for better model performance.

### 7.1 Log Transform (Fix right skew)

```python
import numpy as np
import matplotlib.pyplot as plt

# Skewed data (like income, house prices)
data = np.array([100, 200, 300, 1000, 5000, 50000])

log_data = np.log1p(data)   # log1p = log(1+x), handles 0 safely

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.hist(data, bins=10); plt.title("Original (Skewed)")
plt.subplot(1,2,2); plt.hist(log_data, bins=10); plt.title("Log Transformed")
plt.show()
```

### 7.2 Square Root Transform

```python
sqrt_data = np.sqrt(data)
print(sqrt_data)
# Milder than log — good for count data (Poisson distributed)
```

### 7.3 Box-Cox Transform (Auto finds best power)

```python
from scipy.stats import boxcox

# Only works on positive values
data_positive = [100, 200, 300, 1000, 5000]
transformed, lambda_val = boxcox(data_positive)
print(f"Lambda: {lambda_val:.4f}")
print(transformed)
```

### 7.4 Yeo-Johnson Transform (Works with negatives too)

```python
from sklearn.preprocessing import PowerTransformer

data = np.array([[-1, 2], [0.5, 3], [0.8, -2], [1.2, 5]])

pt = PowerTransformer(method='yeo-johnson')
transformed = pt.fit_transform(data)
print(transformed)
```

### 7.5 Quantile Transform (Force normal distribution)

```python
from sklearn.preprocessing import QuantileTransformer

data = np.array([[1], [2], [3], [100], [500], [5000]])

qt = QuantileTransformer(output_distribution='normal', random_state=42)
transformed = qt.fit_transform(data)
print(transformed)
# Maps to a Gaussian — very powerful but loses interpretability
```

### 7.6 Binarization (Threshold)

```python
from sklearn.preprocessing import Binarizer

data = np.array([[0.5, 1.5, 2.5, 3.5]])

# All values > 2.0 become 1, rest become 0
binarizer = Binarizer(threshold=2.0)
print(binarizer.transform(data))
```

---

## 8. Feature Creation & Interaction Features

> Create new features from existing ones using domain knowledge.

### 8.1 Arithmetic Features

```python
df = pd.DataFrame({
    'length': [10, 20, 15, 30],
    'width': [5, 8, 6, 12],
    'price': [100, 200, 150, 300],
    'quantity': [2, 3, 1, 5]
})

# Create new features
df['area'] = df['length'] * df['width']            # Product
df['aspect_ratio'] = df['length'] / df['width']    # Ratio
df['revenue'] = df['price'] * df['quantity']       # Business logic
df['price_per_area'] = df['price'] / df['area']    # Combined

print(df)
```

### 8.2 Polynomial Features (For linear models)

```python
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

X = np.array([[2, 3], [4, 5]])

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

print("Original features:", X.shape)
print("Polynomial features:", X_poly.shape)
print("Feature names:", poly.get_feature_names_out())
# Creates: x1, x2, x1^2, x1*x2, x2^2
```

### 8.3 Interaction Features (Cross features)

```python
df = pd.DataFrame({
    'age': [25, 35, 45, 55],
    'income': [30000, 60000, 90000, 120000],
    'gender': ['M', 'F', 'M', 'F']
})

# Interaction: age × income
df['age_income'] = df['age'] * df['income']

# Ratio features
df['income_per_age'] = df['income'] / df['age']

# Cross features (categorical × numerical)
df['gender_income'] = df['gender'] + '_' + df['income'].astype(str)

print(df)
```

### 8.4 Aggregation Features (Group statistics)

```python
df = pd.DataFrame({
    'customer_id': [1, 1, 1, 2, 2, 3],
    'purchase_amount': [100, 200, 150, 300, 50, 400],
    'category': ['A', 'B', 'A', 'C', 'A', 'B']
})

# Group-level statistics (very powerful for tabular data)
agg = df.groupby('customer_id')['purchase_amount'].agg(
    total_spend='sum',
    avg_spend='mean',
    max_spend='max',
    min_spend='min',
    num_purchases='count',
    std_spend='std'
).reset_index()

print(agg)
# Merge back to original
df = df.merge(agg, on='customer_id', how='left')
```

### 8.5 Domain-Specific Features (Examples)

```python
# E-commerce example
df = pd.DataFrame({
    'views': [100, 200, 50, 300],
    'clicks': [10, 40, 5, 90],
    'purchases': [2, 8, 1, 15],
    'revenue': [200, 1600, 50, 3000]
})

df['ctr'] = df['clicks'] / df['views']                     # Click-Through Rate
df['conversion_rate'] = df['purchases'] / df['clicks']     # Conversion Rate
df['avg_order_value'] = df['revenue'] / df['purchases']    # AOV
df['revenue_per_view'] = df['revenue'] / df['views']       # Efficiency

print(df)
```

---

## 9. Date & Time Features

> Temporal features are gold for forecasting, recommendation, and fraud detection.

### 9.1 Extract Basic Components

```python
import pandas as pd

df = pd.DataFrame({
    'timestamp': pd.to_datetime([
        '2024-01-15 09:30:00',
        '2024-07-04 14:15:00',
        '2023-12-25 22:00:00',
        '2024-03-08 06:45:00'
    ])
})

# Extract all useful time components
df['year'] = df['timestamp'].dt.year
df['month'] = df['timestamp'].dt.month
df['day'] = df['timestamp'].dt.day
df['hour'] = df['timestamp'].dt.hour
df['minute'] = df['timestamp'].dt.minute
df['day_of_week'] = df['timestamp'].dt.dayofweek      # 0=Monday
df['day_name'] = df['timestamp'].dt.day_name()
df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
df['quarter'] = df['timestamp'].dt.quarter
df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)

print(df.T)
```

### 9.2 Time Since / Time Delta Features

```python
reference_date = pd.Timestamp('2024-01-01')

df['days_since_ref'] = (df['timestamp'] - reference_date).dt.days
df['days_since_epoch'] = (df['timestamp'] - pd.Timestamp('1970-01-01')).dt.days

print(df[['timestamp', 'days_since_ref', 'days_since_epoch']])
```

### 9.3 Cyclical Encoding for Time (Very Important!)

```python
import numpy as np

# Problem: Month 12 and Month 1 are CLOSE but numerically far apart
# Solution: Encode cyclically using sin and cos

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)

print(df[['month', 'month_sin', 'month_cos']].head())
```

### 9.4 Lag Features (For time series)

```python
df_ts = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=10),
    'sales': [100, 120, 130, 110, 140, 160, 150, 170, 180, 200]
})

# Lag features — previous values as features
df_ts['sales_lag1'] = df_ts['sales'].shift(1)   # 1 day ago
df_ts['sales_lag7'] = df_ts['sales'].shift(7)   # 7 days ago (last week)

# Rolling window features
df_ts['sales_rolling_mean_3'] = df_ts['sales'].rolling(window=3).mean()
df_ts['sales_rolling_std_3'] = df_ts['sales'].rolling(window=3).std()
df_ts['sales_rolling_max_7'] = df_ts['sales'].rolling(window=7).max()

# Expanding window (cumulative stats)
df_ts['sales_cumulative_mean'] = df_ts['sales'].expanding().mean()

print(df_ts)
```

### 9.5 Business Day Features

```python
from pandas.tseries.offsets import BDay
import numpy as np

df['is_business_day'] = df['timestamp'].apply(
    lambda x: 1 if x.weekday() < 5 else 0
)

# Days until end of month
df['days_until_month_end'] = df['timestamp'].apply(
    lambda x: (x + pd.offsets.MonthEnd(0) - x).days
)

print(df[['timestamp', 'is_business_day', 'days_until_month_end']])
```

---

## 10. Text Feature Engineering

> Transform raw text into numbers that capture meaning.

### 10.1 Basic Text Cleaning

```python
import re
import string

def clean_text(text):
    text = text.lower()                                    # Lowercase
    text = re.sub(r'http\S+|www\S+', '', text)            # Remove URLs
    text = re.sub(r'<.*?>', '', text)                      # Remove HTML tags
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)                        # Remove numbers
    text = re.sub(r'\s+', ' ', text).strip()               # Remove extra spaces
    return text

texts = [
    "Hello World! Visit https://example.com for more.",
    "This is GREAT!!! 123 amazing products."
]

cleaned = [clean_text(t) for t in texts]
print(cleaned)
```

### 10.2 Tokenization & Stopword Removal

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

text = "the quick brown fox jumps over the lazy dog"

# Tokenize
tokens = word_tokenize(text)

# Remove stopwords
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w not in stop_words]

print("Tokens:", tokens)
print("Filtered:", filtered)
```

### 10.3 Stemming & Lemmatization

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ['running', 'runs', 'ran', 'easily', 'fairly', 'studies', 'studied']

print("Stemming (aggressive, may not be real word):")
print([stemmer.stem(w) for w in words])

print("\nLemmatization (real base form):")
print([lemmatizer.lemmatize(w) for w in words])
```

### 10.4 Bag of Words (BoW)

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love deep learning"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print("Vocabulary:", vectorizer.vocabulary_)
print("BoW Matrix:\n", X.toarray())
print("Feature names:", vectorizer.get_feature_names_out())
```

### 10.5 TF-IDF (Term Frequency - Inverse Document Frequency)

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF = how important a word is to a document in a collection
# High TF-IDF = word appears often in this doc but rarely in others

vectorizer = TfidfVectorizer(max_features=20, ngram_range=(1, 2))
X = vectorizer.fit_transform(corpus)

import pandas as pd
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
print(tfidf_df)
```

### 10.6 N-grams

```python
# Unigrams, Bigrams, Trigrams
vectorizer = CountVectorizer(ngram_range=(1, 3))  # 1-gram to 3-gram
X = vectorizer.fit_transform(corpus)
print("Features:", vectorizer.get_feature_names_out())
```

### 10.7 Word Embeddings (Word2Vec)

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "machine", "learning"],
    ["machine", "learning", "is", "amazing"],
    ["deep", "learning", "is", "powerful"]
]

# Train Word2Vec
model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, epochs=10)

# Get word vector
print("Vector for 'machine':", model.wv['machine'][:5])

# Find similar words
print("Similar to 'learning':", model.wv.most_similar('learning'))

# Document embedding = average of word vectors
def doc_embedding(tokens):
    vecs = [model.wv[w] for w in tokens if w in model.wv]
    return sum(vecs) / len(vecs) if vecs else [0]*100
```

### 10.8 Sentence Transformers (Best for NLP/Gen AI)

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Machine learning is fascinating",
    "AI is changing the world",
    "I like pizza"
]

# Encode sentences to dense vectors
embeddings = model.encode(sentences)
print("Shape:", embeddings.shape)  # (3, 384)

# Compute similarity
from sklearn.metrics.pairwise import cosine_similarity
sim = cosine_similarity(embeddings)
print("Similarity matrix:\n", sim)
```

### 10.9 Statistical Text Features

```python
df = pd.DataFrame({'text': [
    "This is great!",
    "I absolutely LOVE this product. Best purchase ever!!!",
    "bad"
]})

# Basic statistical features
df['char_count'] = df['text'].str.len()
df['word_count'] = df['text'].str.split().str.len()
df['avg_word_length'] = df['text'].apply(
    lambda x: sum(len(w) for w in x.split()) / len(x.split())
)
df['num_exclamations'] = df['text'].str.count('!')
df['num_uppercase'] = df['text'].apply(lambda x: sum(1 for c in x if c.isupper()))
df['uppercase_ratio'] = df['num_uppercase'] / df['char_count']

print(df)
```

---

## 11. Image Feature Engineering

### 11.1 Raw Pixel Features

```python
from PIL import Image
import numpy as np

# Load and flatten image
img = Image.open('image.jpg').resize((64, 64)).convert('L')  # Grayscale
pixel_array = np.array(img).flatten()   # 64*64 = 4096 features
print("Pixel features shape:", pixel_array.shape)
```

### 11.2 HOG Features (Histogram of Oriented Gradients)

```python
from skimage.feature import hog
from skimage import io, color

img = io.imread('image.jpg')
img_gray = color.rgb2gray(img)

features, hog_image = hog(img_gray, 
                           orientations=8, 
                           pixels_per_cell=(16, 16),
                           cells_per_block=(1, 1),
                           visualize=True)

print("HOG features shape:", features.shape)
```

### 11.3 CNN Feature Extraction (Transfer Learning)

```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Use pretrained ResNet to extract features
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last FC layer
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img = Image.open('image.jpg')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    features = model(tensor).squeeze().numpy()

print("CNN features shape:", features.shape)  # (2048,) for ResNet50
```

---

## 12. Feature Selection

> More features ≠ better model. Too many features = overfitting, slow training.

### 12.1 Filter Methods (Statistics-based)

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression, 
    chi2, mutual_info_classif
)
from sklearn.datasets import load_iris
import pandas as pd

X, y = load_iris(return_X_y=True)

# Select top K features by ANOVA F-score (classification)
selector = SelectKBest(score_func=f_classif, k=2)
X_selected = selector.fit_transform(X, y)
print("Selected features shape:", X_selected.shape)

# Get scores
scores = pd.Series(selector.scores_, index=[f'feature_{i}' for i in range(X.shape[1])])
print(scores.sort_values(ascending=False))
```

### 12.2 Variance Threshold (Remove constant features)

```python
from sklearn.feature_selection import VarianceThreshold

X = [[0, 2, 0, 3],
     [0, 1, 4, 3],
     [0, 1, 1, 3]]

# Remove features with variance < 0.1 (nearly constant)
selector = VarianceThreshold(threshold=0.1)
X_filtered = selector.fit_transform(X)
print("Filtered shape:", X_filtered.shape)
```

### 12.3 Correlation Analysis

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(100, 5), columns=list('ABCDE'))

corr_matrix = df.corr()

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()

# Remove highly correlated features (> 0.9)
def remove_correlated(df, threshold=0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=to_drop)

df_reduced = remove_correlated(df)
print("Original:", df.shape, "Reduced:", df_reduced.shape)
```

### 12.4 Wrapper Methods (RFE)

```python
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)

# Recursive Feature Elimination — tries all subsets
model = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model, n_features_to_select=10)
rfe.fit(X, y)

print("Selected features mask:", rfe.support_)
print("Feature ranking:", rfe.ranking_)
```

### 12.5 Embedded Methods (Feature Importance from Models)

```python
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

# Feature importance from tree models
importances = pd.Series(rf.feature_importances_).sort_values(ascending=False)
importances.plot(kind='bar')
plt.title("Feature Importances")
plt.show()

# Select top 10 features
top_features = importances.head(10).index
X_top = X[:, top_features]
```

### 12.6 LASSO Regularization (L1 — zeros out features)

```python
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)

# Coefficients = 0 means feature not useful
coeff = pd.Series(lasso.coef_).sort_values(ascending=False)
selected = coeff[coeff != 0]
print(f"LASSO selected {len(selected)} / {X.shape[1]} features")
print(selected)
```

### 12.7 SHAP Values (Explainable AI Feature Importance)

```python
# pip install shap
import shap
from sklearn.ensemble import GradientBoostingClassifier

model = GradientBoostingClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

# Global feature importance
shap.summary_plot(shap_values, X)

# Single prediction explanation
shap.force_plot(explainer.expected_value, shap_values[0], X[0])
```

---

## 13. Dimensionality Reduction

> Reduce number of features while preserving information.

### 13.1 PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# PCA to 2 components
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Total variance explained:", sum(pca.explained_variance_ratio_))

# Visualize
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1'); plt.ylabel('PC2')
plt.title('PCA of Iris Dataset')
plt.show()

# Choose n_components to explain 95% variance
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X_scaled)
print(f"Components for 95% variance: {pca_95.n_components_}")
```

### 13.2 t-SNE (Visualization, non-linear)

```python
from sklearn.manifold import TSNE

# t-SNE — great for visualization, NOT for feature engineering pipeline
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='Set1')
plt.title('t-SNE Visualization')
plt.show()

# ⚠️ t-SNE is for visualization only — not for preprocessing
```

### 13.3 UMAP (Better than t-SNE, faster)

```python
# pip install umap-learn
import umap

reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='Set1')
plt.title('UMAP Visualization')
plt.show()
```

### 13.4 Truncated SVD (For sparse data / text)

```python
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = ["I love ML", "ML is great", "I love AI", "AI is the future"]

# TF-IDF then SVD = Latent Semantic Analysis (LSA)
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(corpus)  # sparse matrix

svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_tfidf)
print("SVD output shape:", X_svd.shape)
```

### 13.5 Autoencoders (Deep Learning based reduction)

```python
import tensorflow as tf
from tensorflow import keras

# Define autoencoder
input_dim = 30
encoding_dim = 10

input_layer = keras.Input(shape=(input_dim,))
encoded = keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = keras.Model(input_layer, decoded)
encoder = keras.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

X, _ = load_breast_cancer(return_X_y=True)
X_scaled = MinMaxScaler().fit_transform(X)

autoencoder.fit(X_scaled, X_scaled, epochs=50, batch_size=16, verbose=0)

# Extract encoded features
X_encoded = encoder.predict(X_scaled)
print("Encoded features shape:", X_encoded.shape)  # (569, 10)
```

---

## 14. Handling Imbalanced Datasets

> Imbalanced = one class has far more samples (e.g., fraud: 99% legit, 1% fraud).

### 14.1 Understand the Problem

```python
from sklearn.datasets import make_classification
import pandas as pd

X, y = make_classification(n_samples=1000, weights=[0.95, 0.05], random_state=42)

print("Class distribution:")
print(pd.Series(y).value_counts())
# Class 0: 950, Class 1: 50  ← imbalanced!
```

### 14.2 Oversampling — SMOTE

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("After SMOTE:")
print(pd.Series(y_resampled).value_counts())
# SMOTE creates SYNTHETIC minority samples (not duplicates)
```

### 14.3 Undersampling

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

print("After Undersampling:")
print(pd.Series(y_resampled).value_counts())
```

### 14.4 Combine Both — SMOTETomek

```python
from imblearn.combine import SMOTETomek

smt = SMOTETomek(random_state=42)
X_resampled, y_resampled = smt.fit_resample(X, y)
print(pd.Series(y_resampled).value_counts())
```

### 14.5 Class Weights (Easiest fix!)

```python
from sklearn.linear_model import LogisticRegression

# Tell model to penalize minority class errors more
model = LogisticRegression(class_weight='balanced')
model.fit(X, y)
# 'balanced' automatically adjusts weights based on class frequency
```

### 14.6 Use Right Metrics for Imbalanced Data

```python
from sklearn.metrics import (classification_report, roc_auc_score,
                              precision_recall_curve, average_precision_score)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# ❌ DON'T USE: Accuracy (misleading for imbalanced)
# ✅ USE:
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
print(f"Average Precision: {average_precision_score(y_test, y_prob):.4f}")
```

---

## 15. Feature Engineering for Deep Learning

### 15.1 Embedding Layers (Replace OHE for high cardinality)

```python
import tensorflow as tf
from tensorflow import keras

# Instead of one-hot encoding 1000 categories → use embedding
n_categories = 1000
embedding_dim = 32

model = keras.Sequential([
    keras.layers.Embedding(input_dim=n_categories, output_dim=embedding_dim),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
```

### 15.2 Batch Normalization (In-network feature scaling)

```python
model = keras.Sequential([
    keras.layers.Dense(128),
    keras.layers.BatchNormalization(),    # Normalize inside network
    keras.layers.Activation('relu'),
    keras.layers.Dense(64),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
```

### 15.3 Feature Hashing for Neural Nets

```python
from tensorflow.keras.layers import Hashing, Embedding

# Hash categorical IDs to fixed size
hasher = Hashing(num_bins=512)
embedded_layer = Embedding(input_dim=512, output_dim=16)
```

### 15.4 Tabular Deep Learning (Wide & Deep)

```python
# Wide = memorization (linear model)
# Deep = generalization (neural net)
# Combined = powerful for tabular data

wide_inputs = keras.Input(shape=(100,))      # One-hot features
deep_inputs = keras.Input(shape=(10,))       # Dense features

# Wide part
wide = keras.layers.Dense(1)(wide_inputs)

# Deep part
deep = keras.layers.Dense(64, activation='relu')(deep_inputs)
deep = keras.layers.Dense(32, activation='relu')(deep)
deep = keras.layers.Dense(1)(deep)

# Combined
output = keras.layers.Add()([wide, deep])
output = keras.layers.Activation('sigmoid')(output)

model = keras.Model(inputs=[wide_inputs, deep_inputs], outputs=output)
model.summary()
```

---

## 16. Feature Engineering for NLP & Gen AI

### 16.1 Chunking for RAG (Retrieval Augmented Generation)

```python
# Text chunking is FEATURE ENGINEERING for Gen AI
def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for RAG"""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

long_text = "..." * 1000  # Long document
chunks = chunk_text(long_text, chunk_size=100, overlap=20)
print(f"Created {len(chunks)} chunks")
```

### 16.2 Metadata as Features for RAG

```python
# For each document chunk, add metadata features
def create_rag_features(chunk, doc_metadata):
    return {
        'text': chunk,
        'source': doc_metadata['source'],
        'page_number': doc_metadata['page'],
        'doc_type': doc_metadata['type'],
        'date': doc_metadata['date'],
        'word_count': len(chunk.split()),
        'char_count': len(chunk),
        'has_code': int('```' in chunk or 'def ' in chunk),
        'has_table': int('|' in chunk),
        'language': detect_language(chunk)  # e.g., using langdetect
    }
```

### 16.3 Prompt Engineering as Feature Engineering

```python
# System prompt = feature for LLM behavior
# Few-shot examples = feature templates

def create_prompt_features(query, context_chunks, few_shot_examples):
    """Feature engineering for LLM prompts"""
    
    # Feature 1: Retrieved context
    context = "\n\n".join([f"[Source {i+1}]: {c}" for i, c in enumerate(context_chunks)])
    
    # Feature 2: Few-shot examples
    shots = "\n".join([f"Q: {q}\nA: {a}" for q, a in few_shot_examples])
    
    # Feature 3: Structured prompt
    prompt = f"""You are a helpful assistant.

Context:
{context}

Examples:
{shots}

Question: {query}
Answer:"""
    
    return prompt
```

### 16.4 Embedding Features for Vector Stores

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks mimic the brain"
]

# Generate embeddings = features for vector store
embeddings = model.encode(documents)
print("Embeddings shape:", embeddings.shape)

# Store in vector database (e.g., FAISS)
import faiss

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))

# Query
query = "What is deep learning?"
query_embedding = model.encode([query])
distances, indices = index.search(query_embedding.astype('float32'), k=2)
print("Top results:", [documents[i] for i in indices[0]])
```

### 16.5 Token Features for Fine-tuning

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

text = "Feature engineering is crucial for ML"

# Tokenize — this IS feature engineering for transformers
tokens = tokenizer(
    text,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print("Input IDs:", tokens['input_ids'])
print("Attention Mask:", tokens['attention_mask'])
print("Token Type IDs:", tokens['token_type_ids'])
```

---

## 17. Feature Engineering for Agentic AI

### 17.1 State Features for Agents

```python
# In agentic AI, state = features the agent observes
def create_agent_state_features(raw_state):
    """Convert raw environment state to agent-friendly features"""
    return {
        # Task features
        'task_complexity': estimate_complexity(raw_state['task']),
        'task_type': classify_task(raw_state['task']),
        'subtask_count': len(raw_state.get('subtasks', [])),
        
        # Context features
        'available_tools': raw_state.get('tools', []),
        'tool_count': len(raw_state.get('tools', [])),
        'has_retrieval': int('search' in raw_state.get('tools', [])),
        'has_code_exec': int('code' in raw_state.get('tools', [])),
        
        # Memory features
        'context_length': len(raw_state.get('history', '')),
        'steps_taken': raw_state.get('step_count', 0),
        'tokens_used': raw_state.get('token_count', 0),
        
        # Progress features
        'completion_pct': raw_state.get('progress', 0),
        'errors_encountered': raw_state.get('error_count', 0)
    }
```

### 17.2 Tool Selection Features

```python
# Feature engineering for tool routing
def create_tool_routing_features(query):
    """Create features to decide which tool to use"""
    return {
        'needs_search': int(any(w in query.lower() for w in 
                               ['recent', 'latest', 'current', 'news', '2024', '2025'])),
        'needs_math': int(any(w in query.lower() for w in 
                             ['calculate', 'compute', 'how many', 'total'])),
        'needs_code': int(any(w in query.lower() for w in 
                             ['code', 'script', 'program', 'function'])),
        'needs_retrieval': int(any(w in query.lower() for w in 
                                  ['document', 'file', 'pdf', 'report'])),
        'query_length': len(query.split()),
        'is_question': int(query.strip().endswith('?')),
    }
```

### 17.3 Memory & Conversation Features

```python
# Feature engineering for conversation history
def extract_conversation_features(history):
    """Turn conversation history into agent features"""
    messages = history.get('messages', [])
    
    return {
        'turn_count': len(messages),
        'user_messages': sum(1 for m in messages if m['role'] == 'user'),
        'assistant_messages': sum(1 for m in messages if m['role'] == 'assistant'),
        'avg_user_msg_length': sum(len(m['content'].split()) 
                                   for m in messages if m['role'] == 'user') / max(1, len(messages)),
        'topics_discussed': extract_topics([m['content'] for m in messages]),
        'sentiment_trend': analyze_sentiment_trend(messages),
        'unresolved_questions': count_unresolved(messages)
    }
```

---

## 18. Feature Stores & MLOps

> Feature stores = centralized systems to store, serve, and reuse features.

### 18.1 Why Feature Stores?

```
Problem: 
- Data scientists compute same features repeatedly
- Training features ≠ serving features (training-serving skew)
- Features not reusable across teams

Solution: Feature Store
```

### 18.2 Feast (Open-source Feature Store)

```python
# pip install feast
from feast import FeatureStore, Entity, Feature, FeatureView, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define entities
customer = Entity(name="customer_id", description="Customer ID")

# Define feature source
customer_source = FileSource(
    path="data/customer_features.parquet",
    timestamp_field="event_timestamp"
)

# Define feature view
customer_fv = FeatureView(
    name="customer_features",
    entities=["customer_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="total_spend", dtype=Float32),
        Feature(name="num_orders", dtype=Int64),
        Feature(name="avg_rating", dtype=Float32),
    ],
    source=customer_source
)

# Use the store
store = FeatureStore(repo_path="feature_repo/")
store.apply([customer, customer_fv])

# Retrieve features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=["customer_features:total_spend", "customer_features:num_orders"]
).to_df()
```

### 18.3 Feature Pipeline with Sklearn Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

# Separate feature types
numerical_features = ['age', 'salary', 'experience']
categorical_features = ['city', 'department']

# Preprocessing for numerical
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_features),
    ('cat', categorical_transformer, categorical_features)
])

# Full pipeline with model
full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Fit and predict
import pandas as pd
import numpy as np

X = pd.DataFrame({
    'age': [25, np.nan, 35, 45],
    'salary': [50000, 60000, np.nan, 80000],
    'experience': [2, 5, 8, 15],
    'city': ['Delhi', 'Mumbai', np.nan, 'Chennai'],
    'department': ['IT', 'HR', 'Finance', 'IT']
})
y = [0, 1, 0, 1]

full_pipeline.fit(X, y)
print("Pipeline trained successfully!")
```

---

## 19. Automated Feature Engineering (AutoFE)

### 19.1 Featuretools (Deep Feature Synthesis)

```python
# pip install featuretools
import featuretools as ft
import pandas as pd

# Create dataset
customers = pd.DataFrame({
    'customer_id': [1, 2, 3],
    'age': [25, 35, 45],
    'city': ['Delhi', 'Mumbai', 'Chennai']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105],
    'customer_id': [1, 1, 2, 3, 3],
    'amount': [100, 200, 150, 300, 250],
    'date': pd.to_datetime(['2024-01-01', '2024-02-01', '2024-01-15', 
                            '2024-03-01', '2024-03-15'])
})

# Define EntitySet
es = ft.EntitySet(id='ecommerce')
es = es.add_dataframe(dataframe_name='customers', dataframe=customers, index='customer_id')
es = es.add_dataframe(dataframe_name='orders', dataframe=orders, index='order_id',
                       time_index='date')
es = es.add_relationship('customers', 'customer_id', 'orders', 'customer_id')

# Deep Feature Synthesis — automatically creates hundreds of features!
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name='customers',
    max_depth=2
)

print("Generated features:", feature_matrix.shape[1])
print(feature_matrix.head())
```

### 19.2 AutoSklearn / H2O AutoML (End-to-end)

```python
# H2O AutoML handles FE + model selection automatically
import h2o
from h2o.automl import H2OAutoML

h2o.init()

df = h2o.import_file('data.csv')
train, test = df.split_frame(ratios=[0.8])

x = train.columns[:-1]
y = train.columns[-1]

aml = H2OAutoML(max_models=20, seed=42)
aml.train(x=x, y=y, training_frame=train)

print(aml.leaderboard)
best_model = aml.leader
```

---

## 20. End-to-End Pipeline Example

> Complete feature engineering pipeline from raw data to model predictions.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# STEP 1: Load Data
# ─────────────────────────────────────────
np.random.seed(42)
n = 1000

df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'salary': np.random.exponential(50000, n),
    'experience': np.random.randint(0, 40, n),
    'city': np.random.choice(['Delhi', 'Mumbai', 'Chennai', None], n, p=[0.3, 0.3, 0.3, 0.1]),
    'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
    'signup_date': pd.date_range('2020-01-01', periods=n, freq='D'),
    'target': np.random.randint(0, 2, n)
})

# Introduce missing values
df.loc[df.sample(frac=0.1).index, 'age'] = np.nan
df.loc[df.sample(frac=0.1).index, 'salary'] = np.nan

print("Dataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())

# ─────────────────────────────────────────
# STEP 2: Feature Engineering
# ─────────────────────────────────────────

# Date features
df['signup_year'] = df['signup_date'].dt.year
df['signup_month'] = df['signup_date'].dt.month
df['days_since_signup'] = (pd.Timestamp.now() - df['signup_date']).dt.days
df['signup_month_sin'] = np.sin(2 * np.pi * df['signup_month'] / 12)
df['signup_month_cos'] = np.cos(2 * np.pi * df['signup_month'] / 12)

# Missing indicator
df['age_missing'] = df['age'].isnull().astype(int)
df['salary_missing'] = df['salary'].isnull().astype(int)

# Interaction features
df['salary_per_year'] = df['salary'] / (df['experience'] + 1)
df['senior'] = (df['age'] > 45).astype(int)

# Ordinal encoding for education
edu_map = {'High School': 0, 'Bachelor': 1, 'Master': 2, 'PhD': 3}
df['education_ord'] = df['education'].map(edu_map)

# ─────────────────────────────────────────
# STEP 3: Prepare for Modeling
# ─────────────────────────────────────────

features = ['age', 'salary', 'experience', 'city', 'education_ord',
            'signup_year', 'days_since_signup', 'signup_month_sin',
            'signup_month_cos', 'age_missing', 'salary_missing',
            'salary_per_year', 'senior']

X = df[features]
y = df['target']

numerical_cols = ['age', 'salary', 'experience', 'education_ord',
                  'signup_year', 'days_since_signup', 'signup_month_sin',
                  'signup_month_cos', 'salary_per_year']
categorical_cols = ['city']
passthrough_cols = ['age_missing', 'salary_missing', 'senior']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ─────────────────────────────────────────
# STEP 4: Build Pipeline
# ─────────────────────────────────────────

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, numerical_cols),
    ('cat', cat_pipeline, categorical_cols),
    ('pass', 'passthrough', passthrough_cols)
])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('selector', SelectKBest(f_classif, k=10)),
    ('model', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# ─────────────────────────────────────────
# STEP 5: Train & Evaluate
# ─────────────────────────────────────────

model_pipeline.fit(X_train, y_train)
y_pred = model_pipeline.predict(X_test)
y_prob = model_pipeline.predict_proba(X_test)[:, 1]

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

cv_scores = cross_val_score(model_pipeline, X, y, cv=5, scoring='roc_auc')
print(f"CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 21. Interview Questions & Cheatsheet

### Top Interview Questions

**Q1: What's the difference between normalization and standardization?**
> Normalization scales to [0,1] using min-max. Standardization scales to mean=0, std=1. Use normalization when you know bounds; use standardization when data is normally distributed or for linear/neural models.

**Q2: When do you use median vs mean imputation?**
> Mean for symmetric distributions. Median for skewed data or when outliers exist (median is robust to outliers).

**Q3: Why is one-hot encoding bad for high cardinality?**
> Creates too many sparse columns (curse of dimensionality). Use target encoding, binary encoding, or embeddings instead.

**Q4: What is target encoding and its risk?**
> Replace category with mean of target. Risk = data leakage if not done correctly (fit on train, transform test separately with smoothing).

**Q5: Decision trees don't need feature scaling — why?**
> Trees split on thresholds, not distances or gradients. Scale doesn't affect split decisions.

**Q6: What is feature leakage?**
> Using information in features that wouldn't be available at prediction time. E.g., using future data to predict the past.

**Q7: How do you handle time-series features?**
> Lag features, rolling statistics, cyclical encoding, no random shuffle (maintain time order).

**Q8: What's the difference between PCA and feature selection?**
> PCA creates NEW features (combinations). Feature selection picks EXISTING features. PCA loses interpretability.

---

### 🎯 Quick Reference Cheatsheet

```
MISSING VALUES
├── < 5%  → Mean/Median/Mode
├── 5-30% → KNN/Iterative Imputer  
├── 30-50%→ Add missing indicator
└── > 50% → Drop column

ENCODING
├── 2 categories        → Label Encoding
├── < 10 categories     → One-Hot Encoding
├── 10-100 categories   → Binary Encoding
├── > 100 categories    → Target/Frequency Encoding
└── NLP/DL              → Embeddings

SCALING
├── Linear/Logistic/SVM/KNN/NN → StandardScaler
├── With outliers               → RobustScaler
├── Need [0,1] range            → MinMaxScaler
└── Tree-based models           → No scaling needed

OUTLIERS
├── Remove  → Only if < 1%, genuine errors
├── Cap     → Winsorize to IQR boundaries
├── Transform → Log/sqrt reduces impact
└── Flag    → Add outlier indicator column

IMBALANCED DATA
├── Easy fix  → class_weight='balanced'
├── Oversample→ SMOTE (create synthetic minority)
├── Undersample→ Only if lots of majority data
└── Metrics   → ROC-AUC, Precision-Recall, F1

FEATURE SELECTION
├── Filter    → Correlation, Chi2, ANOVA (fast)
├── Wrapper   → RFE (best but slow)
├── Embedded  → LASSO, Tree importance (best tradeoff)
└── Explain   → SHAP values (interpretable)

DIMENSIONALITY REDUCTION
├── PCA   → Linear, interpretable, fast
├── t-SNE → Visualization only
├── UMAP  → Better t-SNE, preserves global structure
└── Autoencoder → Non-linear, deep learning
```

---

## 📌 Libraries Reference

```python
# Core
import pandas as pd
import numpy as np

# Sklearn (main FE library)
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder,
    PolynomialFeatures, PowerTransformer, QuantileTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.feature_selection import (
    SelectKBest, RFE, VarianceThreshold,
    f_classif, f_regression, mutual_info_classif
)
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Advanced encoding
import category_encoders as ce   # pip install category_encoders

# Imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

# Auto FE
import featuretools as ft          # pip install featuretools

# NLP
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import shap                        # pip install shap

# DL
import tensorflow as tf
import torch
from transformers import AutoTokenizer

# Time series
# (built into pandas — .dt accessor, .shift(), .rolling())
```

---

*📌 These notes cover everything from beginner to advanced Feature Engineering for ML, DL, NLP, Gen AI, and Agentic AI. Save this file and use it as your reference guide.*

*🔖 Star this file if it helped! Happy Learning! 🚀*
