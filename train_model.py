import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

# loading dataset
df = pd.read_csv(
    "data/CarPrice_Assignment.csv"
)

print(df.head())

# checking shape
print("\nDataset Shape :")
print(df.shape)

# checking missing values
print("\nMissing Values :")
print(df.isnull().sum())

# removing duplicate rows
df.drop_duplicates(inplace=True)

# removing columns that are not useful
# car_ID is just serial numbering
# CarName has too many unique values

df.drop(
    columns=["car_ID", "CarName"],
    inplace=True,
    errors="ignore"
)

# encoding categorical columns
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

# correlation matrix
plt.figure(figsize=(14,10))

sns.heatmap(
    df.corr(),
    cmap="coolwarm"
)

plt.title("Correlation Matrix")

plt.show()

# selecting important features
selected_features = [

    "symboling",

    "fueltype",

    "aspiration",

    "doornumber",

    "horsepower",

    "peakrpm",

    "citympg",

    "highwaympg",

    "enginesize",

    "curbweight",

    "wheelbase"
]

X = df[selected_features]

y = df["price"]

# splitting dataset
X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42
)

# =========================================
# PRE PRUNING
# =========================================

# limiting tree growth before training

pre_pruning_params = {

    "max_depth": [3,5,7,10],

    "min_samples_split": [2,5,10],

    "min_samples_leaf": [1,2,4]
}

grid_search = GridSearchCV(

    DecisionTreeRegressor(),

    pre_pruning_params,

    cv=5,

    scoring="r2"
)

grid_search.fit(X_train, y_train)

pre_pruning_model = grid_search.best_estimator_

pre_pred = pre_pruning_model.predict(
    X_test
)

pre_r2 = r2_score(
    y_test,
    pre_pred
)

pre_mae = mean_absolute_error(
    y_test,
    pre_pred
)

print("\nPre Pruning R2 Score :")
print(pre_r2)

print("\nPre Pruning MAE :")
print(pre_mae)

print("\nBest Parameters :")
print(grid_search.best_params_)

# =========================================
# POST PRUNING
# =========================================

# first training a fully grown tree

dt = DecisionTreeRegressor(
    random_state=42
)

dt.fit(X_train, y_train)

# getting pruning path
path = dt.cost_complexity_pruning_path(
    X_train,
    y_train
)

ccp_alphas = path.ccp_alphas

models = []

# training trees with different alpha values

for alpha in ccp_alphas:

    model = DecisionTreeRegressor(
        ccp_alpha=alpha
    )

    model.fit(X_train, y_train)

    models.append(model)

scores = []

for model in models:

    y_pred = model.predict(X_test)

    score = r2_score(
        y_test,
        y_pred
    )

    scores.append(score)

# selecting best pruned tree

best_model_index = scores.index(
    max(scores)
)

post_pruning_model = models[
    best_model_index
]

post_pred = post_pruning_model.predict(
    X_test
)

post_r2 = r2_score(
    y_test,
    post_pred
)

post_mae = mean_absolute_error(
    y_test,
    post_pred
)

print("\nPost Pruning R2 Score :")
print(post_r2)

print("\nPost Pruning MAE :")
print(post_mae)

# saving models

pickle.dump(

    pre_pruning_model,

    open(
        "models/prepruning_model.pkl",
        "wb"
    )
)

pickle.dump(

    post_pruning_model,

    open(
        "models/postpruning_model.pkl",
        "wb"
    )
)

print("\nModels Saved Successfully")