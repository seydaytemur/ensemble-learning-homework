import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

def plot_model_comparison(results):
    """
    model comparison
    """
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    ax = sns.barplot(x=names, y=accuracies, palette="viridis")
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.4f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha = 'center', va = 'center', 
                    xytext = (0, 9), 
                    textcoords = 'offset points',
                    fontsize=12, fontweight='bold')

    plt.title('Modellerin Doğruluk (Accuracy) Karşılaştırması', fontsize=16)
    plt.ylabel('Doğruluk Oranı', fontsize=12)
    plt.ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def plot_confusion_matrices(results, y_test):
    """
    her model için confusion matrix 
    """
    n_models = len(results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    
    if n_models == 1:
        axes = [axes]
        
    for i, (name, res) in enumerate(results.items()):
        cm = confusion_matrix(y_test, res['predictions'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Tahmin Edilen')
        axes[i].set_ylabel('Gerçek Değer')
        
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

def plot_comprehensive_metrics(results):
    """
    metrics(Precision, Recall, F1-Score)
    """
    metrics_list = []
    
    for name, res in results.items():
        # Ağırlıklı ortalamaları alıyoruz (weighted avg)
        w_avg = res['report_dict']['weighted avg']
        metrics_list.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': w_avg['precision'],
            'Recall': w_avg['recall'],
            'F1-Score': w_avg['f1-score']
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # table
    print("\n" + "="*70)
    print("DETAYLI PERFORMANS METRİKLERİ")
    print("="*70)
    print(df_metrics.to_string(index=False))
    print("="*70)
    
    # (Heatmap) metrics table
    plt.figure(figsize=(10, 5))
    df_plot = df_metrics.set_index('Model')
    
    sns.heatmap(df_plot, annot=True, cmap="YlGnBu", fmt=".4f", cbar=True)
    plt.title('Modeller Arası Detaylı Metrik Karşılaştırması', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('detailed_metrics.png')
    plt.close()
    
    print("[VİDEO/GÖRSEL] Tüm metriklerin olduğu karşılaştırma tablosu 'detailed_metrics.png' olarak kaydedildi.")
