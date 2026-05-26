import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score

# loading dataset
df = pd.read_csv(
    "data/CarPrice_Assignment.csv"
)

print(df.head())

# =========================
# FEATURE ENGINEERING
# =========================

# removing unnecessary columns
df.drop(
    columns=["car_ID", "CarName"],
    inplace=True
)

# label encoding categorical columns
label_encoder = LabelEncoder()

categorical_cols = [
    "fueltype",
    "aspiration",
    "doornumber",
    "carbody",
    "drivewheel",
    "enginelocation",
    "enginetype",
    "cylindernumber",
    "fuelsystem"
]

for col in categorical_cols:

    df[col] = label_encoder.fit_transform(
        df[col]
    )

# =========================
# CORRELATION MATRIX
# =========================

plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# =========================
# SELECTING INPUT & OUTPUT
# =========================

X = df.drop("price", axis=1)

y = df["price"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# HYPERPARAMETER TUNING
# =========================

param_grid = {
    "max_depth": [3,5,7,10],
    "min_samples_split": [2,5,10],
    "min_samples_leaf": [1,2,4]
}

grid_search = GridSearchCV(
    DecisionTreeRegressor(),
    param_grid,
    cv=5,
    scoring="r2"
)

grid_search.fit(X_train, y_train)

# best model
model = grid_search.best_estimator_

# predictions
y_pred = model.predict(X_test)

# model score
r2 = r2_score(y_test, y_pred)

print("R2 Score :", r2)

print("Best Parameters :")

print(grid_search.best_params_)

# saving model
pickle.dump(
    model,
    open("models/dtr_model.pkl", "wb")
)

print("Model Saved Successfully")