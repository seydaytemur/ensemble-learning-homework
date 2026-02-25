import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from data_preprocessing import preprocess_data
from visualization import plot_model_comparison, plot_confusion_matrices

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} Eğitiliyor ---")
        # Model training
        model.fit(X_train, y_train)
        
        # Prediction
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_str = classification_report(y_test, y_pred)
        
        results[name] = {
            'accuracy': acc,
            'report_dict': report_dict,
            'report_str': report_str,
            'predictions': y_pred
        }
        
        print(f"Doğruluk: {acc:.4f}")
        print("Sınıflandırma Raporu:")
        print(report_str)
        
    return results

if __name__ == "__main__":
    # Data preprocessing
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'Heart_Disease_Prediction.csv')
    
    print("Data preprocessing starts...")
    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    
    
    # 1. Bagging example: Random Forest
    # 2. Boosting example: AdaBoost
    # 3. Modern Boosting example: XGBoost
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
    }
    
    # Train models and get results
    all_results = train_and_evaluate(models, X_train, X_test, y_train, y_test)
    
    # Summary comparison
    print("\n" + "="*30)
    print("MODEL COMPARISON SUMMARY")
    print("="*30)
    for name, res in all_results.items():
        print(f"{name:40} | Doğruluk: {res['accuracy']:.4f}")
    print("="*30)

    # visualization
    print("\nSonuçlar görselleştiriliyor...")
    plot_model_comparison(all_results)
    plot_confusion_matrices(all_results, y_test)
    
    # comprehensive metrics table and graph
    from visualization import plot_comprehensive_metrics
    plot_comprehensive_metrics(all_results)

    print("\nİpucu: 'model_comparison.png', 'confusion_matrices.png' ve 'detailed_metrics.png' dosyalarını kontrol edebilirsiniz.")
