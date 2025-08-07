import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- Load Dataset ---
df = pd.read_csv("heart.csv")
print("Initial Data Preview:\n", df)

# --- Check for Missing Values ---
print("\nMissing Values:\n", df.isnull().sum())

# --- Encode Categorical Columns ---
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# --- Visualizations ---

# Separate numeric and categorical columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Histograms
df[numeric_cols].hist(figsize=(15, 10), bins=20)
plt.suptitle('Histograms of Numeric Features')
plt.tight_layout()
plt.show()

# Bar Chart for Heart Disease
plt.figure(figsize=(6, 4))
sns.countplot(x='HeartDisease', data=df)
plt.title("Bar Chart - HeartDisease Count")
plt.show()

# Pie Chart for Gender
plt.figure(figsize=(5, 5))
gender_counts = df['Sex'].value_counts()
plt.pie(gender_counts, labels=['Male', 'Female'], autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart - Gender Distribution")
plt.axis('equal')
plt.show()

# Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", square=True)
plt.title('Correlation Heatmap')
plt.show()

# --- Feature Scaling ---
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# --- Logistic Regression with Class Balance ---
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

log_acc = accuracy_score(y_test, log_preds)
print("\nLogistic Regression Accuracy:", round(log_acc * 100, 4), "%")

# --- Optional: Random Forest Model for Comparison ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

rf_acc = accuracy_score(y_test, rf_preds)
print("Random Forest Accuracy:", round(rf_acc * 100, 4), "%")

# --- Confusion Matrix (for better model) ---
best_model_name = "Random Forest" if rf_acc > log_acc else "Logistic Regression"
best_preds = rf_preds if rf_acc > log_acc else log_preds

cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title(f"{best_model_name} - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Classification Report ---
print(f"\nClassification Report ({best_model_name}):\n", classification_report(y_test, best_preds))
