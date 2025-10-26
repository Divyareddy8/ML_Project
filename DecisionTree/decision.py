import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import warnings

warnings.filterwarnings("ignore")

# ==============================
# LOAD DATA
# ==============================
train = pd.read_csv("../train.csv")
test = pd.read_csv("../test.csv")

test_ids = test['Hospital_Id']
train.drop_duplicates(inplace=True)

# ==============================
# FEATURE ENGINEERING
# ==============================
for df in [train, test]:
    df['Order_Placed_Date'] = pd.to_datetime(df['Order_Placed_Date'], errors='coerce')
    df['Delivery_Date'] = pd.to_datetime(df['Delivery_Date'], errors='coerce')
    df['Delivery_Lag_Days'] = (df['Delivery_Date'] - df['Order_Placed_Date']).dt.days.fillna(0).astype(int)
    df['Order_Day_of_Week'] = df['Order_Placed_Date'].dt.dayofweek
    df['Order_Month'] = df['Order_Placed_Date'].dt.month.fillna(0).astype(int)
    df['Equipment_Volume'] = df['Equipment_Height'] * (df['Equipment_Width'] + 1e-6)
    df['Equipment_Density'] = df['Equipment_Weight'] / df['Equipment_Volume']
    df['Value_per_Weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + 1e-6)
    df['Fee_per_Weight'] = df['Base_Transport_Fee'] / (df['Equipment_Weight'] + 1e-6)
    df['Urgent_and_Fragile'] = (df['Urgent_Shipping'].map({'Yes': 1, 'No': 0}) * df['Fragile_Equipment'].map({'Yes': 1, 'No': 0}))

# ==============================
# COLUMN DEFINITIONS
# ==============================
numeric_cols = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight', 'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Lag_Days', 'Order_Day_of_Week', 'Order_Month', 'Equipment_Volume', 'Equipment_Density', 'Value_per_Weight', 'Fee_per_Weight', 'Urgent_and_Fragile']
binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 'Fragile_Equipment', 'Rural_Hospital']
categorical_cols = ['Equipment_Type', 'Transport_Method', 'Hospital_Info']
cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']

# ==============================
# HANDLING MISSING DATA
# ==============================
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='Other')
train[numeric_cols] = num_imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = num_imputer.transform(test[numeric_cols])
for col in categorical_cols:
    train[col] = cat_imputer.fit_transform(train[[col]]).ravel()
    test[col] = cat_imputer.transform(test[[col]]).ravel()

# ==============================
# BINARY MAPPING
# ==============================
def map_binary(series): return series.astype(str).str.lower().isin(['yes', 'y', 'true', '1']).astype(int)
for df in [train, test]:
    for col in binary_cols: df[col] = map_binary(df[col])

# ==============================
# RARE CATEGORY HANDLING
# ==============================
freq_threshold = 0.05
for col in categorical_cols:
    rare = train[col].value_counts(normalize=True)[lambda x: x < freq_threshold].index
    train[col] = train[col].replace(rare, 'Other')
    test[col] = test[col].replace(rare, 'Other')

# ==============================
# DROP UNUSED COLUMNS
# ==============================
train.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')
test.drop([c for c in cols_to_drop if c != 'Hospital_Id'], axis=1, inplace=True, errors='ignore')

# ==============================
# TARGET + SPLIT
# ==============================
y = np.log(train['Transport_Cost'].clip(lower=1))
X = train.drop('Transport_Cost', axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# PREPROCESSORS (Renamed & Single Line)
# ==============================
num_preproc = Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())])
cat_preproc = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Other')), ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
full_preproc = ColumnTransformer([('num', num_preproc, numeric_cols + binary_cols), ('cat', cat_preproc, categorical_cols)])

# ==============================
# UTILITY FUNCTION
# ==============================
def evaluate_and_save(model_name, model, X_val, y_val, X_test, file_name):
    y_pred_val = np.exp(model.predict(X_val)); y_true_val = np.exp(y_val)
    mse = mean_squared_error(y_true_val, y_pred_val); rmse = np.sqrt(mse); r2 = r2_score(y_true_val, y_pred_val)
    print(f"{model_name} → MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")
    y_test_pred = np.exp(model.predict(X_test)).clip(0, None)
    pd.DataFrame({'Hospital_Id': test_ids, 'Transport_Cost': y_test_pred}).to_csv(file_name, index=False)
    print(f"{model_name} predictions saved to '{file_name}'\n")
    return mse, rmse, r2

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, KFold

# ==============================
# DECISION TREE GRIDSEARCH
# ==============================
dt_pipe = Pipeline([
    ('preprocessor', full_preproc),
    ('regressor', DecisionTreeRegressor(random_state=42))
])

param_grid_dt = {
    'regressor__max_depth': [5, 10, 15, 20, None],
    'regressor__min_samples_split': [2, 5, 10, 20],
    'regressor__min_samples_leaf': [1, 2, 5, 10]
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

dt_grid = GridSearchCV(
    dt_pipe,
    param_grid=param_grid_dt,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=-1
)

dt_grid.fit(X_train, y_train)

# Best parameters
best_dt_params = dt_grid.best_params_
print(f"Best Decision Tree parameters: {best_dt_params}\n")

# Fit best model on full training set
dt_best = Pipeline([
    ('preprocessor', full_preproc),
    ('regressor', DecisionTreeRegressor(
        max_depth=best_dt_params['regressor__max_depth'],
        min_samples_split=best_dt_params['regressor__min_samples_split'],
        min_samples_leaf=best_dt_params['regressor__min_samples_leaf'],
        random_state=42
    ))
])

dt_best.fit(X, y)

# Evaluate
mse_dt, rmse_dt, r2_dt = evaluate_and_save(
    "DecisionTree (Tuned)",
    dt_best,
    X_val, y_val,
    test.drop('Hospital_Id', axis=1),
    "DecisionTree_submission_tuned.csv"
)
