# model_training.py
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, save_model
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score  # Importações de métricas de avaliação
import joblib
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.model_selection import KFold
import numpy as np
import os

def create_autoencoder(input_dim, encoding_dim=2):
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    encoder = Dense(16, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)
    
    # Decoder
    decoder = Dense(16, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(decoder)
    decoder = Dense(64, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    
    # Usar otimizador com learning rate ajustado
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder, encoder_model

def train_clustering_models(X_train, n_clusters=3, random_state=42, epochs=50):
    """
    Treina os modelos de clustering (KMeans, DBSCAN e Agglomerative Clustering) e salva-os para uso posterior
    """
    # Criar diretório para salvar os modelos se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Criar e treinar o autoencoder
    input_dim = X_train.shape[1]
    autoencoder, encoder_model = create_autoencoder(input_dim)
    
    # Treinar o autoencoder
    autoencoder.fit(X_train_scaled, X_train_scaled,
                    epochs=epochs,
                    batch_size=32,
                    shuffle=True,
                    verbose=0)
    
    # Obter a representação codificada dos dados
    encoded_data = encoder_model.predict(X_train_scaled)
    
    # Treinar modelos de clustering nos dados codificados
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(encoded_data)
    
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan.fit(encoded_data)
    
    agglomerative = AgglomerativeClustering(n_clusters=n_clusters)
    agglomerative.fit(encoded_data)
    
    # Salvar os modelos
    save_model(encoder_model, 'models/encoder_model')
    joblib.dump(kmeans, 'models/kmeans_model.joblib')
    joblib.dump(dbscan, 'models/dbscan_model.joblib')
    joblib.dump(agglomerative, 'models/agglomerative_model.joblib')
    joblib.dump(scaler, 'models/scaler_model.joblib')
    
    trained_models = {
        "Encoder": encoder_model,
        "KMeans": kmeans,
        "DBSCAN": dbscan,
        "Agglomerative": agglomerative,
        "Scaler": scaler
    }
    
    return trained_models

def find_optimal_clusters(data, max_clusters=10):
    distortions = []
    K = range(1, max_clusters+1)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        distortions.append(kmeans.inertia_)
    
    # Encontrar o "cotovelo" usando método do segundo derivado
    differences = np.diff(distortions)
    second_differences = np.diff(differences)
    optimal_clusters = np.argmin(second_differences) + 2
    
    return optimal_clusters

def preprocess_data(X):
    # Remover outliers
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA para redução inicial de dimensionalidade
    pca = PCA(n_components=0.95)  # Mantém 95% da variância
    X_pca = pca.fit_transform(X_scaled)
    
    return X_pca, scaler, pca

def ensemble_clustering(X, n_clusters):
    results = []
    
    # KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(X)
    results.append(kmeans_labels)
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X)
    results.append(dbscan_labels)
    
    # Agglomerative Clustering
    agglom = AgglomerativeClustering(n_clusters=n_clusters)
    agglom_labels = agglom.fit_predict(X)
    results.append(agglom_labels)
    
    # Consenso final
    consensus_labels = np.zeros(len(X))
    for i in range(len(X)):
        labels = [result[i] for result in results]
        consensus_labels[i] = max(set(labels), key=labels.count)
    
    return consensus_labels

def cross_validate_clustering(X, n_clusters, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        
        # Treinar modelos
        trained_models = train_clustering_models(X_train, n_clusters)
        
        # Avaliar
        labels, _ = predict_clusters(X_val, trained_models)
        score = silhouette_score(X_val, labels)
        scores.append(score)
    
    return np.mean(scores), np.std(scores)


def predict_clusters(X_test, trained_models):
    """
    Faz predições usando os modelos treinados
    """
    scaler = trained_models["Scaler"]
    encoder = trained_models["Encoder"]
    kmeans = trained_models["KMeans"]
    
    # Normalizar os dados de teste
    X_test_scaled = scaler.transform(X_test)
    
    # Codificar os dados
    encoded_test = encoder.predict(X_test_scaled)
    
    # Predizer clusters
    cluster_labels = kmeans.predict(encoded_test)
    
    return cluster_labels, encoded_test

def evaluate_clustering_models(trained_models, X_test):
    evaluation_results = {}
    
    # Encode the test data using the encoder model
    encoder = trained_models["Encoder"]
    encoded_X_test = encoder.predict(X_test)

    for name, model in trained_models.items():
        if isinstance(model, Model):
            continue

        elif hasattr(model, 'predict'):
            # Para KMeans e Agglomerative Clustering
            labels = model.predict(encoded_X_test)
        elif hasattr(model, 'fit_predict'):
            # Para DBSCAN, pois ele usa fit_predict
            labels = model.fit_predict(encoded_X_test)
        else:
            print(f"Skipping {name}: not a valid clustering model.")
            continue

        # Check for unique labels
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            silhouette_avg = silhouette_score(encoded_X_test, labels)
        else:
            silhouette_avg = None
        
        evaluation_results[name] = {
            "silhouette_score": silhouette_avg,
            "unique_labels": unique_labels.tolist()
        }
    return evaluation_results
