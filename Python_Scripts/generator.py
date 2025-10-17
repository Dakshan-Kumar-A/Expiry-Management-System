import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from pymongo import MongoClient
import openpyxl

# MongoDB Connection
client = MongoClient('mongodb://localhost:27017/')
db = client['Data']
raw_collection = db['raw_data']

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define realistic cosmetics data
categories = {
    'Skincare': ['Face Cream', 'Serum', 'Moisturizer', 'Cleanser', 'Toner', 'Face Mask', 'Eye Cream', 'Sunscreen'],
    'Makeup': ['Foundation', 'Lipstick', 'Mascara', 'Eyeliner', 'Blush', 'Eyeshadow', 'Concealer', 'Primer'],
    'Haircare': ['Shampoo', 'Conditioner', 'Hair Oil', 'Hair Serum', 'Hair Mask', 'Hair Spray', 'Hair Gel'],
    'Fragrance': ['Perfume', 'Deodorant', 'Body Spray', 'Cologne'],
    'Body Care': ['Body Lotion', 'Body Wash', 'Body Scrub', 'Hand Cream', 'Foot Cream', 'Body Oil']
}

brands = ['Lakme', 'Maybelline', 'Loreal', 'Olay', 'Neutrogena', 'Garnier', 'Nivea', 'Himalaya', 
          'Biotique', 'Plum', 'Mamaearth', 'WOW', 'Faces Canada', 'Colorbar', 'Sugar']

suppliers = [
    {'name': 'Beauty Supplies India', 'lead_time': 7},
    {'name': 'Cosmetic Wholesale Co', 'lead_time': 10},
    {'name': 'Premium Beauty Distributors', 'lead_time': 5},
    {'name': 'Global Cosmetics Supply', 'lead_time': 14},
    {'name': 'India Beauty Hub', 'lead_time': 8},
    {'name': 'Metro Beauty Supplies', 'lead_time': 6},
    {'name': 'Elite Cosmetics Distributor', 'lead_time': 12}
]

# Shelf life mapping (in days) - realistic expiry times
shelf_life_map = {
    'Face Cream': 730, 'Serum': 365, 'Moisturizer': 730, 'Cleanser': 730, 'Toner': 730,
    'Face Mask': 365, 'Eye Cream': 365, 'Sunscreen': 730,
    'Foundation': 730, 'Lipstick': 730, 'Mascara': 180, 'Eyeliner': 365, 'Blush': 730,
    'Eyeshadow': 730, 'Concealer': 730, 'Primer': 730,
    'Shampoo': 1095, 'Conditioner': 1095, 'Hair Oil': 730, 'Hair Serum': 365,
    'Hair Mask': 365, 'Hair Spray': 1095, 'Hair Gel': 730,
    'Perfume': 1825, 'Deodorant': 1095, 'Body Spray': 1095, 'Cologne': 1825,
    'Body Lotion': 730, 'Body Wash': 730, 'Body Scrub': 365, 'Hand Cream': 730,
    'Foot Cream': 730, 'Body Oil': 730
}

# Generate dataset
data = []
product_id = 1000

for i in range(5000):
    category = random.choice(list(categories.keys()))
    sub_category = random.choice(categories[category])
    brand = random.choice(brands)
    supplier = random.choice(suppliers)
    
    # Product Info
    product_name = f"{brand} {sub_category}"
    
    # Supplier Info
    supplier_id = f"SUP{random.randint(1000, 9999)}"
    supplier_name = supplier['name']
    lead_time = supplier['lead_time']
    
    # Inventory Info
    quantity = random.randint(0, 500)
    reorder_level = random.randint(20, 100)
    reorder_quantity = random.randint(100, 300)
    stock_status = 'In Stock' if quantity > reorder_level else ('Low Stock' if quantity > 0 else 'Out of Stock')
    
    # Pricing Info
    unit_cost = round(random.uniform(50, 2000), 2)
    unit_price = round(unit_cost * random.uniform(1.3, 2.5), 2)
    discount_available = random.choice([True, False])
    discount_percentage = round(random.uniform(5, 40), 2) if discount_available else 0
    gst_percentage = random.choice([12, 18, 28])
    final_price = round(unit_price * (1 - discount_percentage/100) * (1 + gst_percentage/100), 2)
    
    # Date & Expiry Info
    shelf_life = shelf_life_map[sub_category]
    manufacture_date = datetime.now() - timedelta(days=random.randint(0, shelf_life//2))
    expiry_date = manufacture_date + timedelta(days=shelf_life)
    days_to_expiry = (expiry_date - datetime.now()).days

    # Ensure two classes: some items expired, some not
    if i < 50:  # Force first 50 records to be expired
        is_expired = True
        days_to_expiry = -random.randint(1, 30)  # Negative days to expiry
    else:
        is_expired = days_to_expiry < 0
    
    # Sales Info
    units_sold = random.randint(0, 100)
    units_returned = random.randint(0, min(units_sold, 10))
    last_sold_date = datetime.now() - timedelta(days=random.randint(1, 90))
    
    # Analytics
    shelf_life_days = shelf_life
    age_of_stock = (datetime.now() - manufacture_date).days
    profit_per_unit = round(unit_price - unit_cost, 2)
    total_value = round(quantity * unit_price, 2)
    
    # Introduce missing values (NaN, ??) - 3% missing rate
    row = {
        'Product_ID': f'PRD{product_id}',
        'Product_Name': product_name if random.random() > 0.01 else np.nan,
        'Category': category if random.random() > 0.005 else '??',
        'Sub_Category': sub_category,
        'Brand': brand if random.random() > 0.01 else '????',
        'Supplier_ID': supplier_id,
        'Supplier_Name': supplier_name if random.random() > 0.01 else np.nan,
        'Lead_Time_Days': lead_time if random.random() > 0.02 else np.nan,
        'Quantity_in_Stock': quantity if random.random() > 0.01 else np.nan,
        'Reorder_Level': reorder_level,
        'Reorder_Quantity': reorder_quantity,
        'Stock_Status': stock_status,
        'Unit_Cost': unit_cost if random.random() > 0.02 else np.nan,
        'Unit_Price': unit_price,
        'Discount_Available': discount_available,
        'Discount_Percentage': discount_percentage,
        'GST_Percentage': gst_percentage,
        'Final_Price': final_price if random.random() > 0.01 else np.nan,
        'Manufacture_Date': manufacture_date.strftime('%Y-%m-%d') if random.random() > 0.01 else '??',
        'Expiry_Date': expiry_date.strftime('%Y-%m-%d') if random.random() > 0.005 else np.nan,
        'Days_to_Expiry': days_to_expiry if random.random() > 0.01 else np.nan,
        'Is_Expired': is_expired,
        'Units_Sold_Last_Month': units_sold if random.random() > 0.02 else np.nan,
        'Units_Returned': units_returned,
        'Last_Sold_Date': last_sold_date.strftime('%Y-%m-%d'),
        'Shelf_Life_Days': shelf_life_days,
        'Age_of_Stock_Days': age_of_stock if random.random() > 0.02 else np.nan,
        'Profit_Per_Unit': profit_per_unit,
        'Total_Value_in_Stock': total_value
    }
    
    data.append(row)
    product_id += 1

# Create DataFrame
df = pd.DataFrame(data)

# Save to Excel
excel_filename = r'C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Data\cosmetics_raw_data.xlsx'
df.to_excel(excel_filename, index=False)
print(f"✓ Dataset saved to {excel_filename}")

# Save to MongoDB
try:
    # Clear existing data
    raw_collection.delete_many({})
    
    # Convert DataFrame to dictionary and insert
    records = df.to_dict('records')
    raw_collection.insert_many(records)
    print(f"✓ {len(records)} records inserted into MongoDB collection 'raw_data'")
    print(f"✓ Database: Data")
    print(f"✓ Collection: raw_data")
except Exception as e:
    print(f"MongoDB Error: {e}")
    print("Note: Make sure MongoDB is running locally or update connection string")

print("\n" + "="*60)
print("DATASET STATISTICS")
print("="*60)
print(f"Total Records: {len(df)}")
print(f"Total Categories: {df['Category'].nunique()}")
print(f"Total Brands: {df['Brand'].nunique()}")
print(f"Total Products: {len(df)}")
print(f"Missing Values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
print("="*60)