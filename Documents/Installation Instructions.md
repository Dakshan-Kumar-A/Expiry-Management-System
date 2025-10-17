STEP 1: Install Node.js Dependencies
--------------------------------------
Run in terminal:
$ npm install

STEP 2: Install Python Dependencies
--------------------------------------
Run in terminal:
$ pip install pandas numpy scikit-learn xgboost matplotlib seaborn openpyxl pymongo imbalanced-learn python-dotenv

STEP 3: Setup MongoDB
--------------------------------------
Option A - Local MongoDB:
1. Install MongoDB Community Edition from mongodb.com
2. Start MongoDB service:
   - Windows: Start MongoDB service from Services
   - Mac: brew services start mongodb-community
   - Linux: sudo systemctl start mongod

Option B - MongoDB Atlas (Cloud):
1. Create account at mongodb.com/cloud/atlas
2. Create a free cluster
3. Get connection string
4. Update MONGODB_URI in .env file

STEP 4: Setup Email (Gmail example)
--------------------------------------
1. Enable 2-Step Verification in Gmail
2. Generate App Password:
   - Go to Google Account > Security > App passwords
   - Select "Mail" and your device
   - Copy the 16-character password
3. Update EMAIL_USER and EMAIL_PASS in .env

STEP 5: Create Required Directories
--------------------------------------
Create these folders in project root:
- public/
- public/images/
- models/

STEP 6: Run Data Pipeline
--------------------------------------
1. Generate dataset:
   $ python dataset_generator.py

2. Preprocess data:
   $ python data_preprocessing.py

3. Train ML model:
   $ python ml_model_training.py

STEP 7: Start Server
--------------------------------------
Development mode:
$ npm run dev

Production mode:
$ npm start

Server will run on http://localhost:5000

STEP 8: Access Application
--------------------------------------
Admin Dashboard: http://localhost:5000/admin-dashboard.html
Customer Store: http://localhost:5000/customer-store.html
Login Page: http://localhost:5000/login.html
