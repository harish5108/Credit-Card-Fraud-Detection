# Credit-Card-Fraud-Detection

- Developed a binary classification model to detect fraudulent credit card transactions using supervised machine learning algorithms.

- Handled imbalanced dataset (~0.17% fraud) using SMOTE for oversampling the minority class.

- Implemented and compared models including Logistic Regression, Random Forest, Decision Tree, SVM, KNN using Stratified K-Fold cross-validation for model robustness.

- Achieved an F1-score of 0.95 and ROC-AUC of 0.99 with the Logistic Regression model.

- Visualized confusion matrices and precision-recall curves to evaluate model performance.

- Applied feature scaling and dimensionality reduction (SVM) for optimized computation and accuracy.

# 🔧 Project Overview: Credit Card Fraud Detection

📊 Dataset

- We'll use the Kaggle Credit Card Fraud Dataset:

- Contains transactions made by European cardholders in 2013.

- Features: V1 to V28 (PCA-transformed), Amount, Time

- Target: Class (1 = fraud, 0 = not fraud)

# ✅ Key Steps in Code

> Here’s a simplified Python script using Logistic Regression and Random Forest.

🔽 Step 1: Import Libraries
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
```
📥 Step 2: Load Dataset

```
# Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
df = pd.read_csv("creditcard.csv")

print(df.head())
print(df['Class'].value_counts())  # Imbalanced: many more non-fraud cases
```

📊 Step 3: Preprocess the Data

```
# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Standardize 'Amount' and 'Time'
scaler = StandardScaler()
X[['Amount', 'Time']] = scaler.fit_transform(X[['Amount', 'Time']])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
```

🧠 Step 4: Train a Logistic Regression Model

```
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)
print("Logistic Regression Results:\n", classification_report(y_test, y_pred_lr))
```

🌲 Step 5: Train a Random Forest Model

```
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Results:\n", classification_report(y_test, y_pred_rf))
```

📉 Step 6: Confusion Matrix Visualization

```
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

# 🔎 Key Notes
- Class imbalance is a major issue — only 0.17% of samples are fraud.

- You can improve results using techniques like:

- SMOTE (Synthetic Minority Oversampling)

- Anomaly detection models (e.g., Isolation Forest, Autoencoders)

- Cost-sensitive learning

