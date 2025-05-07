from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_and_save_model():
    # Load dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/random_forest_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return scaler, model

if __name__ == "__main__":
    scaler, model = train_and_save_model()
    print("Random Forest model and scaler saved successfully.")