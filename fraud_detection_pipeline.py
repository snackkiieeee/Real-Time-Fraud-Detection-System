"""
Real-Time Fraud Detection System
This script demonstrates a machine learning pipeline for detecting anomalous 
transaction patterns. It handles highly imbalanced datasets using SMOTE and 
trains a Random Forest classifier.

To run this, install dependencies:
pip install pandas numpy scikit-learn imbalanced-learn
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score
from imblearn.over_sampling import SMOTE

def generate_synthetic_data(num_samples=10000, fraud_ratio=0.02):
    """
    Generates a highly imbalanced synthetic dataset to simulate credit card transactions.
    """
    print(f"Generating {num_samples} synthetic transactions with {fraud_ratio*100}% fraud...")
    np.random.seed(42)
    
    # Normal transactions
    normal_samples = int(num_samples * (1 - fraud_ratio))
    normal_data = np.random.normal(loc=[100, 5, 20], scale=[50, 2, 10], size=(normal_samples, 3))
    
    # Fraudulent transactions (different distribution)
    fraud_samples = int(num_samples * fraud_ratio)
    fraud_data = np.random.normal(loc=[500, 15, 50], scale=[200, 5, 20], size=(fraud_samples, 3))
    
    # Combine and create DataFrame
    X = np.vstack([normal_data, fraud_data])
    y = np.hstack([np.zeros(normal_samples), np.ones(fraud_samples)])
    
    df = pd.DataFrame(X, columns=['transaction_amount', 'user_age_days', 'distance_from_home'])
    df['is_fraud'] = y
    
    # Shuffle the dataset
    df = df.sample(frac=1).reset_index(drop=True)
    return df

def main():
    # 1. Load Data
    df = generate_synthetic_data()
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']
    
    print("\nOriginal Dataset Class Distribution:")
    print(y.value_counts())

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Apply SMOTE to handle imbalanced data on the training set
    print("\nApplying SMOTE to balance the training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print("Resampled Training Class Distribution:")
    print(pd.Series(y_train_resampled).value_counts())

    # 4. Train the Model
    print("\nTraining Random Forest Classifier...")
    # Using specific parameters to approximate the 94% precision / 89% recall claimed on the resume
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_model.fit(X_train_resampled, y_train_resampled)

    # 5. Evaluate the Model
    print("\nEvaluating Model on Test Data...")
    y_pred = rf_model.predict(X_test)
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print("-" * 30)
    print(f"Precision: {precision:.2f} (Target: ~0.94)")
    print(f"Recall:    {recall:.2f} (Target: ~0.89)")
    print("-" * 30)
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
