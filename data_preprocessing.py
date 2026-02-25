import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def preprocess_data(file_path):
    
    df = pd.read_csv(file_path)
    
    # (Presence: 1, Absence: 0)
    le = LabelEncoder()
    df['Heart Disease'] = le.fit_transform(df['Heart Disease'])
    
    
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    
    # test & train split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # (Feature Scaling)
    # (Random Forest, XGBoost)
    # best practice olarak ve ileride başka modellerle 
    # karşılaştırma yapılabilmesi adına standartlaştırma yapmamız gerek
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Sütun isimlerini korumak için tekrar DataFrame yapımı
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Heart_Disease_Prediction.csv')
    
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    print("Veri başarıyla işlendi.")
    print(f"Eğitim seti boyutu: {X_train.shape}")
    print(f"Test seti boyutu: {X_test.shape}")
    print("\nİlk 5 satır (Ölçeklendirilmiş):")
    print(X_train.head())
