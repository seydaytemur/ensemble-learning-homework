import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def plot_confusion_matrices(results, y_test, suffix=''):
    """Tüm modeller için karmaşıklık matrislerini tek bir panelde çizer."""
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1: axes = [axes]
    
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name} Confusion Matrix')
        axes[i].set_xlabel('Tahmin')
        axes[i].set_ylabel('Gerçek')
    
    plt.tight_layout()
    plt.savefig(f'karmasiklik_matrisleri{suffix}.png')
    plt.close()

def plot_feature_importance(models, feature_names, suffix='', top_n=10):
    """Her model için (destekleniyorsa) en önemli özellikleri çizer."""
    for name, model in models.items():
        importances = None
        current_features = feature_names
        
        # Random Forest ve AdaBoost için
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        
        # Stacking - Meta-Learner (Logistic Regression) katsayılarını alalım
        elif name == "Stacking" and hasattr(model, 'final_estimator_'):
            if hasattr(model.final_estimator_, 'coef_'):
                # Katsayıların mutlak değerini önem derecesi olarak kabul ediyoruz
                importances = np.abs(model.final_estimator_.coef_[0])
                # Stacking özellikleri base learner isimleridir
                current_features = [est[0] for est in model.estimators]
        
        if importances is not None:
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=importances[indices], y=[current_features[i] for i in indices], palette="magma")
            plt.title(f'{name} - Önem Derecesi / Katsayı Analizi')
            plt.xlabel('Bağıl Önem / Ağırlık')
            plt.tight_layout()
            plt.savefig(f'ozellik_onemi_{name.lower().replace(" ", "_")}{suffix}.png')
            plt.close()
            print(f"{name} için özellik önemi görseli oluşturuldu.")

def plot_pca_variance(pca):
    """PCA açıklanan varyans oranını çizer."""
    plt.figure(figsize=(8, 5))
    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
    plt.axhline(y=0.95, color='r', linestyle='-', label='%95 Sınırı')
    plt.title('PCA - Açıklanan Varyans Analizi')
    plt.xlabel('Bileşen Sayısı')
    plt.ylabel('Kümülatif Varyans')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('pca_varyans_analizi.png')
    plt.close()

def plot_metrics_comparison(all_scenario_data):
    """Farklı senaryoların metriklerini karşılaştıran tablo/grafik çizer."""
    df = pd.DataFrame(all_scenario_data)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Model', y='Accuracy', hue='Senaryo', data=df)
    plt.title('Senaryolara Göre Doğruluk (Accuracy) Karşılaştırması')
    plt.ylim(0.6, 1.0)
    plt.grid(axis='y', alpha=0.3)
    plt.savefig('performans_karsilastirma.png')
    plt.close()

def plot_all_metrics(all_scenario_data):
    """Tüm metrikleri (Acc, Prec, Rec, F1) içeren kapsamlı bir karşılaştırma grafiği çizer."""
    df = pd.DataFrame(all_scenario_data)
    # Metrikleri "melt" ederek görselleştirmeye uygun hale getiriyoruz
    df_melted = df.melt(id_vars=['Senaryo', 'Model'], 
                        value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                        var_name='Metrik', value_name='Skor')
    
    g = sns.catplot(data=df_melted, kind="bar", x="Model", y="Skor", 
                    hue="Metrik", col="Senaryo", palette="muted", height=6, aspect=1)
    
    g.set_axis_labels("", "Skor Değeri")
    g.set_titles("{col_name} Senaryosu")
    g.set(ylim=(0.6, 1.0))
    
    # Skorları çubukların üzerine yazma
    for ax in g.axes.flat:
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=9)

    plt.savefig('tum_metrikler_karsilastirma.png')
    plt.close()

def plot_pca_success_comparison(all_scenario_data):
    """PCA öncesi ve sonrası başarı farkını net bir şekilde karşılaştırır."""
    df = pd.DataFrame(all_scenario_data)
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=df, x="Model", y="Accuracy", hue="Senaryo", palette="coolwarm")
    
    plt.title('PCA İşleminin Model Başarısına Etkisi', fontsize=15, pad=20)
    plt.ylabel('Doğruluk Oranı (Accuracy)', fontsize=12)
    plt.xlabel('Algoritmalar', fontsize=12)
    plt.ylim(0.7, 1.0)
    
    # Çubukların üzerine değerleri yazma
    for p in ax.patches:
        ax.annotate(f'%{p.get_height()*100:.1f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('pca_basari_analizi.png')
    plt.close()
    print("PCA başarı analizi grafiği 'pca_başarı_analizi.png' olarak oluşturuldu.")
