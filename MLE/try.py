import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
import warnings

warnings.filterwarnings("ignore")

# ==============================
# LOAD DATA
# ==============================
# NOTE: Assuming 'train.csv' and 'test.csv' are in the parent directory '../'
try:
    train = pd.read_csv("../train.csv")
    test = pd.read_csv("../test.csv")
except FileNotFoundError:
    print("Ensure 'train.csv' and 'test.csv' are in the correct path ('../').")
    # Placeholder data for demonstration if files are not found (remove in production)
    # raise
    pass

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
    
    # Add small epsilon to avoid division by zero/near zero
    epsilon = 1e-6
    df['Equipment_Volume'] = df['Equipment_Height'] * (df['Equipment_Width'] + epsilon)
    df['Equipment_Density'] = df['Equipment_Weight'] / (df['Equipment_Volume'] + epsilon)
    df['Value_per_Weight'] = df['Equipment_Value'] / (df['Equipment_Weight'] + epsilon)
    df['Fee_per_Weight'] = df['Base_Transport_Fee'] / (df['Equipment_Weight'] + epsilon)
    
    # Binary mapping for Urgent and Fragile for the interaction feature
    urgent_map = df['Urgent_Shipping'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    fragile_map = df['Fragile_Equipment'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
    df['Urgent_and_Fragile'] = urgent_map * fragile_map

# ==============================
# COLUMN DEFINITIONS
# ==============================
numeric_cols = ['Supplier_Reliability', 'Equipment_Height', 'Equipment_Width', 'Equipment_Weight',
                'Equipment_Value', 'Base_Transport_Fee', 'Delivery_Lag_Days', 'Order_Day_of_Week',
                'Order_Month', 'Equipment_Volume', 'Equipment_Density', 'Value_per_Weight', 
                'Fee_per_Weight', 'Urgent_and_Fragile']
binary_cols = ['CrossBorder_Shipping', 'Urgent_Shipping', 'Installation_Service', 
               'Fragile_Equipment', 'Rural_Hospital']
categorical_cols = ['Equipment_Type', 'Transport_Method', 'Hospital_Info']
cols_to_drop = ['Hospital_Id', 'Supplier_Name', 'Hospital_Location', 'Order_Placed_Date', 'Delivery_Date']

# ==============================
# HANDLE MISSING DATA (Initial)
# ==============================
num_imputer = SimpleImputer(strategy='median')
cat_imputer = SimpleImputer(strategy='constant', fill_value='Other')

# Impute numeric columns
train[numeric_cols] = num_imputer.fit_transform(train[numeric_cols])
test[numeric_cols] = num_imputer.transform(test[numeric_cols])

# Impute categorical columns
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
test_process = test.drop([c for c in cols_to_drop if c != 'Hospital_Id'], axis=1, errors='ignore')

# ==============================
# TARGET + SPLIT
# ==============================
# Target variable for Gamma GLM MUST be strictly positive (y > 0)
y = train['Transport_Cost'].clip(lower=1e-3)
X = train.drop('Transport_Cost', axis=1)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================
# PREPROCESSORS (Full Pipeline)
# ==============================
all_numeric_cols = numeric_cols + binary_cols
num_preproc = Pipeline([('imp', SimpleImputer(strategy='median')), 
                        ('scl', StandardScaler())])
cat_preproc = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='Other')), 
                        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
full_preproc = ColumnTransformer([('num', num_preproc, all_numeric_cols), 
                                  ('cat', cat_preproc, categorical_cols)], 
                                  remainder='passthrough')

# ==============================
# TRANSFORM DATA
# ==============================
X_train_t = full_preproc.fit_transform(X_train)
X_val_t = full_preproc.transform(X_val)
X_test_t = full_preproc.transform(test_process.drop('Hospital_Id', axis=1, errors='ignore'))

# Handle NaNs/Infs in the transformed features
X_train_t = np.nan_to_num(X_train_t, nan=0.0, posinf=1e6, neginf=-1e6)
X_val_t = np.nan_to_num(X_val_t, nan=0.0, posinf=1e6, neginf=-1e6) # FIXED: Replaced U+00A0 with standard space
X_test_t = np.nan_to_num(X_test_t, nan=0.0, posinf=1e6, neginf=-1e6) # FIXED: Replaced U+00A0 with standard space

# Add constant for intercept (Crucial Fix for statsmodels GLM)
X_train_sm = sm.add_constant(X_train_t, prepend=True) 
X_val_sm = sm.add_constant(X_val_t, prepend=True) # FIXED: Replaced U+00A0 with standard space
X_test_sm = sm.add_constant(X_test_t, prepend=True) # FIXED: Replaced U+00A0 with standard space

print(f"Shape of training matrix for GLM (with intercept): {X_train_sm.shape}")

# ==============================
# TRAIN GAMMA GLM (MLE)
# ==============================
gamma_model = sm.GLM(y_train, X_train_sm, family=sm.families.Gamma(link=sm.genmod.families.links.log()))
gamma_res = gamma_model.fit(maxiter=200)
print(gamma_res.summary())

# ==============================
# EVALUATE
# ==============================
# Use the constant-added validation matrix (X_val_sm)
y_val_pred = gamma_res.predict(X_val_sm)

# Ensure predictions are positive before evaluation
y_val_pred = np.maximum(y_val_pred, 1e-3)

mse = mean_squared_error(y_val, y_val_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_val_pred)
print(f"\nGamma GLM → MSE: {mse:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

# ==============================
# SAVE SUBMISSION
# ==============================
# Use the constant-added test matrix (X_test_sm)
y_pred = gamma_res.predict(X_test_sm)

# Final predictions must be non-negative
y_pred = np.maximum(y_pred, 0) 

submission = pd.DataFrame({'Hospital_Id': test_ids, 'Transport_Cost': y_pred})
submission.to_csv("GammaGLM_submission.csv", index=False)
print("\nGamma GLM predictions saved to 'GammaGLM_submission.csv'")