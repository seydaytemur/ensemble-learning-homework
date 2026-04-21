import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from data_preprocessing import preprocess_data
import visualization as viz

def run_experiment():
    data_path = 'Heart_Disease_Prediction.csv'
    all_metrics = []
    
    # 1. SENARYO: BASELINE (Ham Veri + Varsayılan Parametreler)
    print("\n[1] Baseline Analizi Başlıyor...")
    X_train, X_test, y_train, y_test, _, feature_names = preprocess_data(data_path, use_pca=False)
    
    base_models = [
        ('rf', RandomForestClassifier(random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('ada', AdaBoostClassifier(random_state=42))
    ]
    
    models = {
        "Random Forest": RandomForestClassifier(random_state=42),
        "AdaBoost": AdaBoostClassifier(random_state=42),
        "Stacking": StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)
    }
    
    baseline_results = {}
    for name, model in models.items():
        # Hold-out eğitimi
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Cross-Validation (Overfitting kontrolü için)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        
        baseline_results[name] = {'predictions': y_pred}
        
        all_metrics.append({
            'Senaryo': 'Baseline',
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'CV_Mean': cv_scores.mean()
        })
    
    # Baseline Görselleri
    viz.plot_confusion_matrices(baseline_results, y_test, suffix='_baseline')
    viz.plot_feature_importance(models, feature_names, suffix='_baseline')
    
    # 2. SENARYO: ADVANCED (PCA + Optimizasyon)
    print("[2] Gelişmiş Analiz (PCA + Optimizasyon) Başlıyor...")
    X_train_pca, X_test_pca, y_train, y_test, pca_obj, _ = preprocess_data(data_path, use_pca=True)
    viz.plot_pca_variance(pca_obj)
    
    # Basit bir grid search (Stacking meta-learner için)
    param_grid_stacking = {'final_estimator__C': [0.1, 1.0, 10.0]}
    gs_stacking = GridSearchCV(models["Stacking"], param_grid_stacking, cv=5)
    gs_stacking.fit(X_train_pca, y_train)
    
    advanced_models = {
        "Random Forest": models["Random Forest"].fit(X_train_pca, y_train),
        "AdaBoost": models["AdaBoost"].fit(X_train_pca, y_train),
        "Stacking": gs_stacking.best_estimator_
    }
    
    advanced_results = {}
    for name, model in advanced_models.items():
        y_pred = model.predict(X_test_pca)
        report = classification_report(y_test, y_pred, output_dict=True)
        cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5)
        
        advanced_results[name] = {'predictions': y_pred}
        
        all_metrics.append({
            'Senaryo': 'Advanced (PCA)',
            'Model': name,
            'Accuracy': report['accuracy'],
            'Precision': report['weighted avg']['precision'],
            'Recall': report['weighted avg']['recall'],
            'F1-Score': report['weighted avg']['f1-score'],
            'CV_Mean': cv_scores.mean()
        })
        
    # Gelişmiş Görseller
    viz.plot_confusion_matrices(advanced_results, y_test, suffix='_advanced')
    viz.plot_metrics_comparison(all_metrics)
    viz.plot_all_metrics(all_metrics)
    viz.plot_pca_success_comparison(all_metrics)
    
    # Gelişmiş Özellik Önemi (PCA Bileşenleri için)
    pca_feature_names = [f"Bileşen {i+1}" for i in range(X_train_pca.shape[1])]
    viz.plot_feature_importance(advanced_models, pca_feature_names, suffix='_advanced')
    
    # SONUÇLARIN KAYDEDİLMESİ
    df_metrics = pd.DataFrame(all_metrics)
    with open("final_sonuclar.txt", "w", encoding="utf-8") as f:
        f.write("=== TÜM MODEL VE SENARYO SONUÇLARI ===\n\n")
        f.write(df_metrics.to_string(index=False))
        f.write("\n\nNot: CV_Mean değeri 5-katlı çapraz doğrulama sonucudur. ")
        f.write("Hold-out Accuracy ile CV_Mean birbirine yakınsa model genellenebilir demektir.")
    
    print("\n[TAMAMLANDI] Tüm analizler bitti, görseller ve 'final_sonuclar.txt' oluşturuldu.")

if __name__ == "__main__":
    run_experiment()
