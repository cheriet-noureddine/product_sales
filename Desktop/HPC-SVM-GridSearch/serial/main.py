import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from grid_search import grid_search

# Load dataset
df = pd.read_csv("data/adults.csv", header=None, na_values="?").dropna()

# Last column is income
y = df.iloc[:, -1].apply(lambda v: 1 if ">50K" in str(v) else 0)

X = df.iloc[:, :-1]

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include=["object", "string"]).columns
encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
X_encoded = encoder.fit_transform(X[categorical_cols])

X_numeric = X.drop(columns=categorical_cols).to_numpy(dtype=float)
X_final = np.hstack([X_numeric, X_encoded])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("Train labels distribution:", np.bincount(y_train))
print("Test labels distribution:", np.bincount(y_test))

# Grid search
C_values = [0.1, 1, 10]
gamma_values = ["scale", 0.01, 0.1]

grid_search(X_train, y_train, X_test, y_test, C_values, gamma_values)
