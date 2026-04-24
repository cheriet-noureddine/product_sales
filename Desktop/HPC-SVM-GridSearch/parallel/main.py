import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from grid_search import grid_search_parallel

# Load dataset
data = pd.read_csv("data/adult.csv", header=None, na_values="?").dropna()

X = data.iloc[:, :-1]
y = data.iloc[:, -1].apply(lambda x: 1 if x == ">50K" else 0)

# One-hot encode categorical features
categorical_cols = X.select_dtypes(include=["object"]).columns
encoder = OneHotEncoder(handle_unknown="ignore")
X_encoded = encoder.fit_transform(X[categorical_cols]).toarray()

X_numeric = X.drop(columns=categorical_cols).to_numpy()
X_final = pd.DataFrame(
    data=np.hstack([X_numeric, X_encoded])
)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Grid search parallel
C_values = [0.1, 1, 10]
gamma_values = ["scale", 0.01, 0.1]

grid_search_parallel(X_train, y_train, X_test, y_test, C_values, gamma_values)
