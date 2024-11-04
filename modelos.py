import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import io

def preprocessing(df):
    # Imprimir colunas disponíveis para verificação
    print("Colunas disponíveis:", list(df.columns))
    
    # Selecionar features numéricas apropriadas
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remover colunas que claramente não são úteis para clustering
    features_to_remove = ['game_id', 'game_year']
    features = [col for col in numeric_features if col not in features_to_remove]
    
    # Se 'player_encoded' não existir, usar 'player'
    if 'player_encoded' not in df.columns:
        # Encodificar player se não existir
        label_encoder = LabelEncoder()
        df['player_encoded'] = label_encoder.fit_transform(df['player'])
        features.append('player_encoded')
    
    # Pré-processamento de features
    X = df[features].fillna(0)
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, X, scaler

def dbscan_clustering(X_scaled):
    # Busca de hiperparâmetros
    eps_range = np.linspace(0.1, 1.0, 10)
    min_samples_range = range(2, 6)
    
    best_score = -1
    best_params = None
    best_clusters = None
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            # Ignorar casos sem cluster
            mask = clusters != -1
            if np.sum(mask) > 0:
                try:
                    score = silhouette_score(X_scaled[mask], clusters[mask])
                    
                    if score > best_score:
                        best_score = score
                        best_params = (eps, min_samples)
                        best_clusters = clusters
                except:
                    # Em caso de erro (por exemplo, cluster único)
                    continue
    
    print(f"Melhores parâmetros DBSCAN: eps={best_params[0]}, min_samples={best_params[1]}")
    print(f"Silhouette Score DBSCAN: {best_score}")
    
    return best_clusters

def kmeans_clustering(X_scaled):
    # Método do cotovelo
    inertias = []
    silhouette_scores = []
    k_range = range(2, 10)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        try:
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
        except:
            silhouette_scores.append(-1)  # Valor padrão em caso de erro
    
    # Encontrar o número ideal de clusters
    ideal_k = k_range[np.argmax(silhouette_scores)]
    
    kmeans_final = KMeans(n_clusters=ideal_k, random_state=42, n_init=10)
    final_clusters = kmeans_final.fit_predict(X_scaled)
    
    print(f"Número ideal de clusters K-Means: {ideal_k}")
    print(f"Silhouette Score K-Means: {silhouette_scores[ideal_k-2]}")
    
    return final_clusters

def svm_classification(X_scaled, y):
    # Verificar se y tem variância suficiente
    if len(np.unique(y)) < 2:
        print("Não há variância suficiente em y para classificação.")
        return
    
    # Pipeline para incluir scaler no processo
    pipeline = Pipeline([
        ('svm', SVC(probability=True))
    ])
    
    # Parâmetros para busca
    param_grid = {
        'svm__kernel': ['rbf', 'linear'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': ['scale', 'auto']
    }
    
    # Divisão de dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    # Melhores resultados
    print("Melhores parâmetros SVM:")
    print(grid_search.best_params_)
    
    # Predições com melhor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nRelatório de Classificação SVM:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Matriz de confusão
    print("\nMatriz de Confusão SVM:")
    print(confusion_matrix(y_test, y_pred))

def neural_network_classification(X_scaled, y):
    # Verificar se y tem variância suficiente
    if len(np.unique(y)) < 2:
        print("Não há variância suficiente em y para classificação.")
        return
    
    # Pipeline para incluir scaler
    pipeline = Pipeline([
        ('mlp', MLPClassifier(
            max_iter=2000,  # Aumentar máximo de iterações
            early_stopping=True,  # Parada antecipada
            validation_fraction=0.2,  # Fração para validação
            n_iter_no_change=10  # Número de épocas sem melhora
        ))
    ])
    
    # Parâmetros para busca
    param_grid = {
        'mlp__hidden_layer_sizes': [(10,), (10,5), (20,10)],
        'mlp__activation': ['relu', 'tanh'],
        'mlp__alpha': [0.0001, 0.001, 0.01]
    }
    
    # Divisão de dados
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42
    )
    
    # Grid Search
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='accuracy'
    )
    
    grid_search.fit(X_train, y_train)
    
    # Melhores resultados
    print("Melhores parâmetros Rede Neural:")
    print(grid_search.best_params_)
    
    # Predições com melhor modelo
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\nRelatório de Classificação Rede Neural:")
    print(classification_report(y_test, y_pred, zero_division=1))
    
    # Matriz de confusão
    print("\nMatriz de Confusão Rede Neural:")
    print(confusion_matrix(y_test, y_pred))

def main(data_path):
    # Carregar dados
    try:
        data = pd.read_csv(data_path, sep=',')
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        return
    
    # Pré-processamento
    try:
        X_scaled, X, scaler = preprocessing(data)
    except Exception as e:
        print(f"Erro no pré-processamento: {e}")
        return
    
    # DBSCAN
    print("\n--- DBSCAN Clustering ---")
    try:
        dbscan_clusters = dbscan_clustering(X_scaled)
    except Exception as e:
        print(f"Erro no DBSCAN: {e}")
    
    # K-Means
    print("\n--- K-Means Clustering ---")
    try:
        kmeans_clusters = kmeans_clustering(X_scaled)
    except Exception as e:
        print(f"Erro no K-Means: {e}")
    
    # Preparar dados para classificação
    # Tente usar 'game_grade_encoded' ou outro campo categórico
    target_columns = [
        'game_grade_encoded', 
        'game_grade', 
        'player_encoded'
    ]
    
    # Encontrar primeira coluna válida para classificação
    target_column = None
    for col in target_columns:
        if col in data.columns and len(np.unique(data[col])) > 1:
            target_column = col
            break
    
    if target_column:
        y = data[target_column]
        
        # SVM
        print("\n--- SVM Classification ---")
        svm_classification(X_scaled, y)
        
        # Rede Neural
        print("\n--- Neural Network Classification ---")
        neural_network_classification(X_scaled, y)
    else:
        print("Nenhuma coluna adequada encontrada para classificação.")

# Substitua pelo caminho do seu arquivo CSV
if __name__ == "__main__":
    main("dados.csv")  # Ajuste o caminho conforme necessário