import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    # Load data (skipping the title row)
    df = pd.read_csv('data/bedtime_screentime_sleep_debt.csv', skiprows=1)
    
    # Separate features and target
    X = df.drop(columns=['user_id', 'sleep_debt_category'])
    y_raw = df['sleep_debt_category']
    
    # Encode target variable y into numbers
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw)
    
    # Encode categorical feature columns
    cat_cols = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Model (Random Forest - no external libomp needed!)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    print(f"Model Accuracy: {accuracy_score(y_test, preds):.4f}")
    print(classification_report(y_test, preds))
    
    # Save model and encoders
    joblib.dump(model, 'model.pkl')
    joblib.dump(label_encoders, 'label_encoders.pkl')
    joblib.dump(target_encoder, 'target_encoder.pkl')
    print("Model successfully trained, evaluated, and saved as model.pkl!")

if __name__ == '__main__':
    main()