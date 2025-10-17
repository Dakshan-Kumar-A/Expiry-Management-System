import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_auc_score)
from sklearn.model_selection import cross_val_score, GridSearchCV
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXPIRY PREDICTION MODEL - TRAINING & EVALUATION")
print("="*80)

# STEP 1: Load Training Data
print("\n[1] Loading Training Data...")
X_train = np.load(r"C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Data\X_train.npy")
X_test = np.load(r"C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Data\X_test.npy")
y_train = np.load(r"C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Data\y_train.npy")
y_test = np.load(r"C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Data\y_test.npy")

print(f"   ✓ Training samples: {X_train.shape[0]}")
print(f"   ✓ Testing samples: {X_test.shape[0]}")
print(f"   ✓ Features: {X_train.shape[1]}")

# STEP 2: Model Selection and Training
print("\n[2] Training Multiple Models...")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\n   Training {name}...")
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"   ✓ {name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

# STEP 3: Select Best Model
print("\n[3] Model Selection...")
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"   ✓ Best Model: {best_model_name}")
print(f"   - Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"   - Precision: {results[best_model_name]['precision']:.4f}")
print(f"   - Recall: {results[best_model_name]['recall']:.4f}")
print(f"   - F1 Score: {results[best_model_name]['f1']:.4f}")
print(f"   - ROC-AUC: {results[best_model_name]['roc_auc']:.4f}")
print(f"   - CV Mean: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std']:.4f})")

# STEP 4: Hyperparameter Tuning for Best Model
print("\n[4] Hyperparameter Tuning...")

tuned_model = best_model  # default

if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
elif best_model_name == 'XGBoost':
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'subsample': [0.8, 1.0]
    }
    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
else:
    param_grid = None  # No tuning for Logistic Regression
    base_model = best_model

if param_grid:
    print(f"   Tuning {best_model_name} with GridSearchCV...")
    grid_search = GridSearchCV(base_model, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0)
    grid_search.fit(X_train, y_train)
    tuned_model = grid_search.best_estimator_
    print(f"   ✓ Best Parameters: {grid_search.best_params_}")
else:
    print(f"   ✓ No hyperparameter tuning required for {best_model_name}")

# Re-evaluate tuned model
y_pred_tuned = tuned_model.predict(X_test)
y_pred_proba_tuned = tuned_model.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
tuned_f1 = f1_score(y_test, y_pred_tuned)
tuned_roc_auc = roc_auc_score(y_test, y_pred_proba_tuned)

print(f"   ✓ Tuned Model Performance:")
print(f"     - Accuracy: {tuned_accuracy:.4f}")
print(f"     - F1 Score: {tuned_f1:.4f}")
print(f"     - ROC-AUC: {tuned_roc_auc:.4f}")

# STEP 5: Feature Importance
print("\n[5] Feature Importance Analysis...")
if hasattr(tuned_model, 'feature_importances_'):
    feature_names = [f'Feature_{i}' for i in range(X_train.shape[1])]
    importances = tuned_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("   Top 5 Important Features:")
    for idx, row in feature_importance_df.head(5).iterrows():
        print(f"     {row['Feature']}: {row['Importance']:.4f}")

# STEP 6: Confusion Matrix
print("\n[6] Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred_tuned)
print(f"   True Negatives: {cm[0][0]}")
print(f"   False Positives: {cm[0][1]}")
print(f"   False Negatives: {cm[1][0]}")
print(f"   True Positives: {cm[1][1]}")

# STEP 7: Classification Report
print("\n[7] Detailed Classification Report...")
print(classification_report(y_test, y_pred_tuned, target_names=['Not Expired', 'Expired']))

# STEP 8: Save Model and Artifacts
print("\n[8] Saving Model and Artifacts...")
save1_dir = r"C:\Users\maind\OneDrive\Desktop\Codes\DWDM\Expiry Management System\Model"
os.makedirs(save1_dir, exist_ok=True)

# Save the tuned model
model_path = os.path.join(save1_dir, 'expiry_prediction_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump(tuned_model, f)
print("   ✓ Saved model as 'expiry_prediction_model.pkl'")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'best_params': grid_search.best_params_ if param_grid else None,
    'accuracy': tuned_accuracy,
    'precision': precision_score(y_test, y_pred_tuned),
    'recall': recall_score(y_test, y_pred_tuned),
    'f1_score': tuned_f1,
    'roc_auc': tuned_roc_auc,
    'training_samples': X_train.shape[0],
    'test_samples': X_test.shape[0],
    'features': X_train.shape[1]
}

metadata_path = os.path.join(save1_dir, 'model_metadata.pkl')
with open(metadata_path, 'wb') as f:
    pickle.dump(model_metadata, f)
print("   ✓ Saved model metadata")

# Save feature names
feature_names_list = ['Quantity_in_Stock', 'Days_to_Expiry', 'Units_Sold_Last_Month', 
                      'Age_of_Stock_Days', 'Profit_Margin', 'Stock_Turnover_Ratio',
                      'Expiry_Risk_Encoded', 'Stock_Age_Encoded']
features_path = os.path.join(save1_dir, 'feature_names.pkl')
with open(features_path, 'wb') as f:
    pickle.dump(feature_names_list, f)
print("   ✓ Saved feature names")

# STEP 9: Model Comparison Summary
print("\n[9] Model Comparison Summary...")
print("\n" + "="*80)
print(f"{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
print("="*80)
for name, res in results.items():
    print(f"{name:<25} {res['accuracy']:<12.4f} {res['precision']:<12.4f} {res['recall']:<12.4f} {res['f1']:<12.4f}")
print("="*80)

print("\n✓ TRAINING & EVALUATION COMPLETE ✓")
print("="*80)
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Final Accuracy: {tuned_accuracy:.4f}")
print(f"✓ Final F1-Score: {tuned_f1:.4f}")
print(f"✓ Model saved and ready for deployment")
print("="*80)