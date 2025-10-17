import pickle
import numpy as np
import pandas as pd
from datetime import datetime

class ExpiryPredictor:
    def __init__(self):
        """Initialize the predictor by loading the trained model and metadata"""
        try:
            with open('expiry_prediction_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            with open('model_metadata.pkl', 'rb') as f:
                self.metadata = pickle.load(f)
            
            with open('feature_names.pkl', 'rb') as f:
                self.feature_names = pickle.load(f)
            
            with open('label_encoders.pkl', 'rb') as f:
                self.encoders = pickle.load(f)
            
            print("✓ Model loaded successfully")
            print(f"✓ Model: {self.metadata['model_name']}")
            print(f"✓ Accuracy: {self.metadata['accuracy']:.4f}")
            
        except FileNotFoundError as e:
            print(f"Error: Model files not found. Please train the model first.")
            raise e
    
    def calculate_days_to_expiry(self, expiry_date):
        """Calculate days to expiry from expiry date"""
        if isinstance(expiry_date, str):
            expiry_date = pd.to_datetime(expiry_date)
        return (expiry_date - datetime.now()).days
    
    def calculate_age_of_stock(self, manufacture_date):
        """Calculate age of stock from manufacture date"""
        if isinstance(manufacture_date, str):
            manufacture_date = pd.to_datetime(manufacture_date)
        return (datetime.now() - manufacture_date).days
    
    def calculate_expiry_risk_encoded(self, days_to_expiry):
        """Encode expiry risk level"""
        if days_to_expiry <= 30:
            risk = 'Critical'
        elif days_to_expiry <= 90:
            risk = 'Warning'
        elif days_to_expiry <= 180:
            risk = 'Moderate'
        else:
            risk = 'Safe'
        
        try:
            return self.encoders['risk'].transform([risk])[0]
        except:
            return 2  # Default to moderate
    
    def calculate_stock_age_encoded(self, age_days):
        """Encode stock age category"""
        if age_days <= 30:
            category = 'New'
        elif age_days <= 90:
            category = 'Medium'
        else:
            category = 'Old'
        
        try:
            return self.encoders['stock_age'].transform([category])[0]
        except:
            return 1  # Default to medium
    
    def predict_single_product(self, product_data):
        """
        Predict expiry risk for a single product
        
        Parameters:
        -----------
        product_data : dict
            Dictionary containing product information with keys:
            - quantity_in_stock: int
            - expiry_date: str or datetime
            - manufacture_date: str or datetime
            - units_sold_last_month: int
            - unit_cost: float
            - unit_price: float
        
        Returns:
        --------
        dict with prediction results
        """
        
        # Calculate derived features
        days_to_expiry = self.calculate_days_to_expiry(product_data['expiry_date'])
        age_of_stock = self.calculate_age_of_stock(product_data['manufacture_date'])
        
        # Calculate additional features
        profit_margin = ((product_data['unit_price'] - product_data['unit_cost']) / 
                        product_data['unit_price'] * 100)
        
        stock_turnover = (product_data['units_sold_last_month'] / 
                         (product_data['quantity_in_stock'] + 1))
        
        expiry_risk_encoded = self.calculate_expiry_risk_encoded(days_to_expiry)
        stock_age_encoded = self.calculate_stock_age_encoded(age_of_stock)
        
        # Create feature array in correct order
        features = np.array([[
            product_data['quantity_in_stock'],
            days_to_expiry,
            product_data['units_sold_last_month'],
            age_of_stock,
            profit_margin,
            stock_turnover,
            expiry_risk_encoded,
            stock_age_encoded
        ]])
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        # Determine risk level
        if days_to_expiry <= 0:
            risk_level = "EXPIRED"
            color_code = "red"
        elif days_to_expiry <= 30:
            risk_level = "CRITICAL"
            color_code = "red"
        elif days_to_expiry <= 90:
            risk_level = "WARNING"
            color_code = "orange"
        elif days_to_expiry <= 180:
            risk_level = "MODERATE"
            color_code = "yellow"
        else:
            risk_level = "SAFE"
            color_code = "green"
        
        return {
            'prediction': 'Will Expire Soon' if prediction == 1 else 'Safe',
            'expiry_probability': probability[1],
            'safe_probability': probability[0],
            'days_to_expiry': days_to_expiry,
            'risk_level': risk_level,
            'color_code': color_code,
            'needs_discount': days_to_expiry <= 90 and product_data['quantity_in_stock'] > 0,
            'needs_immediate_action': days_to_expiry <= 30,
            'age_of_stock_days': age_of_stock,
            'stock_turnover_ratio': stock_turnover
        }
    
    def predict_batch(self, products_df):
        """
        Predict expiry risk for multiple products
        
        Parameters:
        -----------
        products_df : pandas.DataFrame
            DataFrame containing product information
        
        Returns:
        --------
        pandas.DataFrame with predictions
        """
        predictions = []
        
        for idx, row in products_df.iterrows():
            try:
                product_data = {
                    'quantity_in_stock': row['Quantity_in_Stock'],
                    'expiry_date': row['Expiry_Date'],
                    'manufacture_date': row['Manufacture_Date'],
                    'units_sold_last_month': row['Units_Sold_Last_Month'],
                    'unit_cost': row['Unit_Cost'],
                    'unit_price': row['Unit_Price']
                }
                
                result = self.predict_single_product(product_data)
                result['Product_ID'] = row.get('Product_ID', idx)
                result['Product_Name'] = row.get('Product_Name', 'Unknown')
                predictions.append(result)
                
            except Exception as e:
                print(f"Error predicting for row {idx}: {e}")
                continue
        
        return pd.DataFrame(predictions)


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("EXPIRY PREDICTION SYSTEM")
    print("="*70)
    
    # Initialize predictor
    predictor = ExpiryPredictor()
    
    # Example 1: Single product prediction
    print("\n[Example 1] Single Product Prediction:")
    print("-" * 70)
    
    sample_product = {
        'quantity_in_stock': 50,
        'expiry_date': '2025-11-15',
        'manufacture_date': '2024-05-01',
        'units_sold_last_month': 20,
        'unit_cost': 150.0,
        'unit_price': 299.0
    }
    
    result = predictor.predict_single_product(sample_product)
    
    print(f"Product Details:")
    print(f"  - Stock: {sample_product['quantity_in_stock']} units")
    print(f"  - Expiry Date: {sample_product['expiry_date']}")
    print(f"  - Price: ₹{sample_product['unit_price']}")
    print(f"\nPrediction Results:")
    print(f"  - Status: {result['prediction']}")
    print(f"  - Risk Level: {result['risk_level']} ({result['color_code']})")
    print(f"  - Days to Expiry: {result['days_to_expiry']}")
    print(f"  - Expiry Probability: {result['expiry_probability']:.2%}")
    print(f"  - Needs Discount: {'Yes' if result['needs_discount'] else 'No'}")
    print(f"  - Immediate Action: {'Required' if result['needs_immediate_action'] else 'Not Required'}")
    
    # Example 2: Batch prediction from CSV
    print("\n[Example 2] Batch Prediction from Database:")
    print("-" * 70)
    
    try:
        # Load some sample data
        df = pd.read_csv('processed_inventory_data.csv').head(10)
        results_df = predictor.predict_batch(df)
        
        print(f"Processed {len(results_df)} products")
        print("\nSummary:")
        print(f"  - Critical Risk: {len(results_df[results_df['risk_level'] == 'CRITICAL'])}")
        print(f"  - Warning: {len(results_df[results_df['risk_level'] == 'WARNING'])}")
        print(f"  - Moderate: {len(results_df[results_df['risk_level'] == 'MODERATE'])}")
        print(f"  - Safe: {len(results_df[results_df['risk_level'] == 'SAFE'])}")
        
        # Save results
        results_df.to_csv('batch_predictions.csv', index=False)
        print(f"\n✓ Results saved to 'batch_predictions.csv'")
        
    except FileNotFoundError:
        print("  Note: No data file found for batch prediction")
    
    print("\n" + "="*70)
    print("PREDICTION SYSTEM READY FOR DEPLOYMENT")
    print("="*70)