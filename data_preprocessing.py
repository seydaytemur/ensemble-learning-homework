import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

def preprocess_data(file_path, use_pca=False, n_components=0.95):
    """
    Veriyi yükler, kodlar, böler ve ölçeklendirir.
    İsteğe bağlı olarak PCA uygular.
    """
    df = pd.read_csv(file_path)
    
    # Hedef Değişkeni Kodlama
    le = LabelEncoder()
    df['Heart Disease'] = le.fit_transform(df['Heart Disease'])
    
    X = df.drop('Heart Disease', axis=1)
    y = df['Heart Disease']
    
    # Eğitim ve Test Ayrımı (Hold-out)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standartlaştırma
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # DataFrame formatına geri dönüş (Sütun isimleri için)
    X_train_final = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_final = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    pca_obj = None
    if use_pca:
        pca_obj = PCA(n_components=n_components)
        X_train_final = pca_obj.fit_transform(X_train_final)
        X_test_final = pca_obj.transform(X_test_final)
        
    return X_train_final, X_test_final, y_train, y_test, pca_obj, X.columns.tolist()
