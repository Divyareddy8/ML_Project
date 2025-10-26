import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
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
    df['Urgent_and_Fragile'] = (df['Urgent_Shipping'].map({'Yes': 1, 'No': 0}) *
                                df['Fragile_Equipment'].map({'Yes': 1, 'No': 0}))

# ==============================
# COLUMN DEFINITIONS
# ==============================
numeric_cols = [
    'Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight',
    'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Lag_Days', 'Order_Day_of_Week',
    'Order_Month', 'Equipment_Volume', 'Equipment_Density', 'Value_per_Weight',
    'Fee_per_Weight', 'Urgent_and_Fragile'
]
binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service',
               'Fragile_Equipment', 'Rural_Hospital']
categorical_cols = ['Equipment_Type', 'Transport_Method', 'Hospital_Info']
cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location',
                'Order_Placed_Date', 'Delivery_Date']

# ==============================
# HANDLE MISSING DATA
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
def map_binary(series):
    return series.astype(str).str.lower().isin(['yes', 'y', 'true', '1']).astype(int)

for df in [train, test]:
    for col in binary_cols:
        df[col] = map_binary(df[col])

# ==============================
# RARE CATEGORY HANDLING
# ==============================
freq_threshold = 0.02
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
# PREPROCESSING PIPELINE
# ==============================
num_preproc = Pipeline([
    ('imp', SimpleImputer(strategy='median')),
    ('scl', StandardScaler())
])
cat_preproc = Pipeline([
    ('imp', SimpleImputer(strategy='constant', fill_value='Other')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

full_preproc = ColumnTransformer([
    ('num', num_preproc, numeric_cols + binary_cols),
    ('cat', cat_preproc, categorical_cols)
])

# ==============================
# TRANSFORM DATA
# ==============================
X_train_proc = full_preproc.fit_transform(X_train)
X_val_proc = full_preproc.transform(X_val)

# ==============================
# MLE REGRESSION IMPLEMENTATION
# ==============================
class MLERegression:
    def __init__(self):
        self.beta = None
        self.sigma = None

    def negative_log_likelihood(self, params, X, y):
        beta = params[:-1]
        sigma = np.exp(params[-1])  # enforce positivity
        residuals = y - X.dot(beta)
        nll = 0.5 * len(y) * np.log(2 * np.pi * sigma**2) + np.sum(residuals**2) / (2 * sigma**2)
        return nll

    def fit(self, X, y):
        init_params = np.zeros(X.shape[1] + 1)
        res = minimize(self.negative_log_likelihood, init_params, args=(X, y), method='BFGS')
        self.beta = res.x[:-1]
        self.sigma = np.exp(res.x[-1])
        return self

    def predict(self, X):
        return X.dot(self.beta)

# ==============================
# FIT & EVALUATE MLE MODEL
# ==============================
mle_model = MLERegression()
mle_model.fit(X_train_proc, y_train)
y_pred = mle_model.predict(X_val_proc)

mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("=== MLE Regression Results ===")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")
print(f"R²: {r2:.4f}")
