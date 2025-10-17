cosmetics-inventory-system/
│
├── server.js                      # Main backend server
├── routes.js                      # API routes continuation
├── package.json                   # Node.js dependencies
├── .env                          # Environment variables
├── requirements.txt              # Python dependencies
│
├── python_scripts/
│   ├── dataset_generator.py      # Generate synthetic data
│   ├── data_preprocessing.py     # Data cleaning & feature engineering
│   ├── ml_model_training.py      # Train ML models
│   └── prediction_script.py      # Make predictions
│
├── models/
│   ├── expiry_prediction_model.pkl
│   ├── model_metadata.pkl
│   ├── feature_names.pkl
│   └── label_encoders.pkl
│
├── data/
│   ├── cosmetics_inventory_raw_data.xlsx
│   ├── processed_inventory_data.csv
│   ├── X_train.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   └── y_test.npy
│
├── public/
│   ├── admin-dashboard.html      # Admin interface
│   ├── customer-store.html       # Customer shopping interface
│   ├── login.html                # Login page
│   ├── register.html             # Registration page
│   ├── checkout.html             # Checkout page
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── app.js
│   └── images/
│       └── default-product.jpg
│
├── documentation/
│   ├── API_Documentation.md
│   ├── User_Manual.pdf
│   ├── System_Architecture.png
│   ├── UML_Diagrams.png
│   └── Project_Report.pdf
│
└── README.md


// ==================== MongoDB Collections Structure ====================


Database: Data

Collections:
1. raw_data        - Original dataset from Excel
2. processed_inventory_data  - Cleaned and processed data
3. users                     - Admin and customer accounts
4. products                  - Active product inventory
5. flashsales                - Flash sale campaigns
6. orders                    - Customer orders
7. tickets                   - Support tickets
8. feedbacks                 - Customer feedback
9. alerts                    - System alerts for admin

