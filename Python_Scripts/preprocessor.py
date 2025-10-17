import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from pymongo import MongoClient
import warnings
warnings.filterwarnings('ignore')

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Data']
raw_collection = db['raw_data']
processed_collection = db['processed_data']

print("="*70)
print("COSMETICS INVENTORY DATA PREPROCESSING PIPELINE")
print("="*70)

# STEP 1: Load Data
print("\n[1] Importing and Inspecting Data...")
try:
    # Try loading from MongoDB first
    data = list(raw_collection.find({}, {'_id': 0}))
    df = pd.DataFrame(data)
    print(f"   ✓ Loaded {len(df)} records from MongoDB")
except:
    # Fallback to Excel
    df = pd.read_excel('cosmetics_inventory_raw_data.xlsx')
    print(f"   ✓ Loaded {len(df)} records from Excel")

print(f"   - Shape: {df.shape}")
print(f"   - Memory Usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# STEP 2: Handle Missing Values
print("\n[2] Handling Missing Values...")
print(f"   Missing values before:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Replace '??' and '????' with NaN
df = df.replace(['??', '????', ''], np.nan)

# Drop rows with missing critical fields (Expiry_Date)
initial_rows = len(df)
df = df.dropna(subset=['Expiry_Date'])
print(f"   ✓ Dropped {initial_rows - len(df)} rows with missing Expiry_Date")

# Fill missing numerical values with median
numerical_cols = ['Lead_Time_Days', 'Quantity_in_Stock', 'Unit_Cost', 'Final_Price', 
                  'Days_to_Expiry', 'Units_Sold_Last_Month', 'Age_of_Stock_Days']
for col in numerical_cols:
    if col in df.columns and df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   ✓ Filled {col} with median: {median_val:.2f}")

# Fill missing categorical values with mode or 'Unknown'
categorical_cols = ['Product_Name', 'Category', 'Brand', 'Supplier_Name', 'Manufacture_Date']
for col in categorical_cols:
    if col in df.columns and df[col].isnull().any():
        if df[col].mode().empty:
            df[col].fillna('Unknown', inplace=True)
        else:
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
        print(f"   ✓ Filled {col} with mode/Unknown")

# STEP 3: Data Type Conversion
print("\n[3] Data Type Conversion...")
# Convert date columns to datetime
date_cols = ['Manufacture_Date', 'Expiry_Date', 'Last_Sold_Date']
for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        print(f"   ✓ Converted {col} to datetime")

# Convert numerical columns
numeric_cols = ['Quantity_in_Stock', 'Unit_Cost', 'Unit_Price', 'Final_Price', 
                'Days_to_Expiry', 'Units_Sold_Last_Month', 'Age_of_Stock_Days']
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# STEP 4: Remove Duplicates
print("\n[4] Removing Duplicates...")
initial_count = len(df)
df = df.drop_duplicates(subset=['Product_ID'], keep='first')
print(f"   ✓ Removed {initial_count - len(df)} duplicate rows")

# STEP 5: Outlier Detection and Treatment
print("\n[5] Outlier Detection and Treatment...")
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    initial = len(df)
    df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    removed = initial - len(df)
    if removed > 0:
        print(f"   ✓ Removed {removed} outliers from {column}")
    return df

# Apply outlier removal to key columns
outlier_cols = ['Unit_Cost', 'Unit_Price', 'Quantity_in_Stock', 'Units_Sold_Last_Month']
for col in outlier_cols:
    if col in df.columns:
        df = remove_outliers_iqr(df, col)

# STEP 6: Feature Engineering
print("\n[6] Feature Engineering...")

# Create new meaningful columns
df['Expiry_Risk_Level'] = df['Days_to_Expiry'].apply(
    lambda x: 'Critical' if x <= 30 else ('Warning' if x <= 90 else ('Moderate' if x <= 180 else 'Safe'))
)
print("   ✓ Created Expiry_Risk_Level")

df['Stock_Turnover_Ratio'] = df['Units_Sold_Last_Month'] / (df['Quantity_in_Stock'] + 1)
print("   ✓ Created Stock_Turnover_Ratio")

df['Revenue_Last_Month'] = df['Units_Sold_Last_Month'] * df['Unit_Price']
print("   ✓ Created Revenue_Last_Month")

df['Stock_Age_Category'] = df['Age_of_Stock_Days'].apply(
    lambda x: 'New' if x <= 30 else ('Medium' if x <= 90 else 'Old')
)
print("   ✓ Created Stock_Age_Category")

df['Profit_Margin'] = ((df['Unit_Price'] - df['Unit_Cost']) / df['Unit_Price'] * 100)
print("   ✓ Created Profit_Margin")

df['Needs_Discount'] = ((df['Days_to_Expiry'] <= 90) & (df['Quantity_in_Stock'] > 0))
print("   ✓ Created Needs_Discount flag")

df['Reorder_Required'] = (df['Quantity_in_Stock'] <= df['Reorder_Level'])
print("   ✓ Created Reorder_Required flag")

# STEP 7: Handling Categorical Data
print("\n[7] Encoding Categorical Data...")

# Label Encoding for ordinal data
le_risk = LabelEncoder()
df['Expiry_Risk_Encoded'] = le_risk.fit_transform(df['Expiry_Risk_Level'])
print(f"   ✓ Label encoded Expiry_Risk_Level: {dict(zip(le_risk.classes_, le_risk.transform(le_risk.classes_)))}")

le_stock_age = LabelEncoder()
df['Stock_Age_Encoded'] = le_stock_age.fit_transform(df['Stock_Age_Category'])
print(f"   ✓ Label encoded Stock_Age_Category")

# One-Hot Encoding for nominal data
df_encoded = pd.get_dummies(df, columns=['Category', 'Stock_Status'], prefix=['Cat', 'Status'])
print(f"   ✓ One-hot encoded Category and Stock_Status")

# STEP 8: Normalization/Scaling
print("\n[8] Normalization and Scaling...")
scaler = StandardScaler()
scale_cols = ['Unit_Cost', 'Unit_Price', 'Quantity_in_Stock', 'Days_to_Expiry', 
              'Units_Sold_Last_Month', 'Age_of_Stock_Days', 'Profit_Margin', 'Stock_Turnover_Ratio']

for col in scale_cols:
    if col in df_encoded.columns:
        df_encoded[f'{col}_Scaled'] = scaler.fit_transform(df_encoded[[col]])
        print(f"   ✓ Scaled {col}")

# STEP 9: Remove Irrelevant Columns
print("\n[9] Removing Irrelevant Columns...")
# Keep original data but mark columns for analysis
irrelevant_cols = ['Product_Name', 'Supplier_Name']  # Keep for reference but not for ML
print(f"   ✓ Identified irrelevant columns for ML: {irrelevant_cols}")

# STEP 10: Data Balancing (for ML classification)
print("\n[10] Data Balancing...")
print(f"   Class distribution for Is_Expired:")
print(df_encoded['Is_Expired'].value_counts())

# Prepare features for SMOTE
ml_features = ['Quantity_in_Stock', 'Days_to_Expiry', 'Units_Sold_Last_Month',
               'Age_of_Stock_Days', 'Profit_Margin', 'Stock_Turnover_Ratio',
               'Expiry_Risk_Encoded', 'Stock_Age_Encoded']

# Check if features exist
ml_features = [f for f in ml_features if f in df_encoded.columns]

X = df_encoded[ml_features]
y = df_encoded['Is_Expired'].astype(int)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
print(f"   ✓ Applied SMOTE")
print(f"   After balancing: {pd.Series(y_balanced).value_counts().to_dict()}")

# STEP 11: Train-Test Split
print("\n[11] Train-Test Split...")
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"   ✓ Training set: {X_train.shape}")
print(f"   ✓ Testing set: {X_test.shape}")

# STEP 12: Save Processed Data
print("\n[12] Saving Processed Data...")

# Save to CSV
df_encoded.to_csv('Data/processed_data.csv', index=False)
print("   ✓ Saved to processed_data.csv")

# Save to MongoDB
try:
    processed_collection.delete_many({})
    
    # Convert datetime to string for MongoDB
    df_mongo = df_encoded.copy()
    for col in df_mongo.select_dtypes(include=['datetime64']).columns:
        df_mongo[col] = df_mongo[col].astype(str)
    
    records = df_mongo.to_dict('records')
    processed_collection.insert_many(records)
    print(f"   ✓ Saved {len(records)} records to MongoDB collection 'processed_data'")
except Exception as e:
    print(f"   ⚠ MongoDB save failed: {e}")

# Save train-test split
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
print("   ✓ Saved train-test split data")

# Save encoders
import pickle
with open('label_encoders.pkl', 'wb') as f:
    pickle.dump({'risk': le_risk, 'stock_age': le_stock_age}, f)
print("   ✓ Saved label encoders")

print("\n" + "="*70)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*70)
print(f"Final Dataset Shape: {df_encoded.shape}")
print(f"Total Features: {len(df_encoded.columns)}")
print(f"Ready for ML Pipeline: ✓")
print("="*70)