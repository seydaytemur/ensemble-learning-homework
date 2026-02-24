import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import preprocess_data

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} Eğitiliyor ---")
        # Modeli eğit
        model.fit(X_train, y_train)
        
        # Tahmin yap
        y_pred = model.predict(X_test)
        
        # Metrikleri hesapla
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': acc,
            'report': report,
            'predictions': y_pred
        }
        
        print(f"Doğruluk: {acc:.4f}")
        print("Sınıflandırma Raporu:")
        print(report)
        
    return results

if __name__ == "__main__":
    # Veriyi önişlemeden geçir
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Heart_Disease_Prediction.csv')
    
    print("Veri önişleme başlıyor...")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    
    # 3 Farklı Topluluk Öğrenmesi Modeli Tanımla
    # 1. Bagging örneği: Random Forest
    # 2. Boosting örneği: AdaBoost
    # 3. Modern Boosting örneği: XGBoost
    models = {
        "Random Forest (Bagging)": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost (Boosting)": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost (Extreme Gradient Boosting)": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    # Modelleri eğit ve sonuçları al
    all_results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    
    # Özet Karşılaştırma
    print("\n" + "="*30)
    print("MODEL KARŞILAŞTIRMA ÖZETİ")
    print("="*30)
    for name, res in all_results.items():
        print(f"{name:40} | Doğruluk: {res['accuracy']:.4f}")
    print("="*30)

    print("\nİpucu: Bu sonuçları makalenizdeki 'Bulgular' bölümüne tablo olarak ekleyebilirsiniz.")
