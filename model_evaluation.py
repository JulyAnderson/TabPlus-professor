from sklearn.metrics import silhouette_score
import numpy as np
from keras.models import Model
from model_training import train_clustering_models  # Ensure this trains without needing y_train
from data_loading_process import load_and_preprocess_data
from sklearn.preprocessing import StandardScaler


#Pré-processamento e carregamento dos dados
df, X_train, X_test, y_train, y_test = load_and_preprocess_data(source ='local')

# Escalonamento dos dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use transform here, not predict

def evaluate_clustering_models(trained_models, X_test):
    evaluation_results = {}
    
    # Encode the test data using the encoder model
    encoder = trained_models["Encoder"]
    encoded_X_test = encoder.predict(X_test)

    for name, model in trained_models.items():
        if isinstance(model, Model):  # Check if it's a Keras model
            # Skip the encoder model as we already encoded the test data
            continue

        elif hasattr(model, 'predict'):  # Check if it's a traditional clustering model
            labels = model.predict(encoded_X_test)  # Use encoded test data

        else:
            print(f"Skipping {name}: not a valid clustering model.")
            continue

        # Check for unique labels
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:  # Ensure at least 2 clusters
            silhouette_avg = silhouette_score(encoded_X_test, labels)  # Use encoded data for silhouette score
        else:
            silhouette_avg = None  # Or some default value like -1
        
        evaluation_results[name] = {
            "silhouette_score": silhouette_avg,
            "unique_labels": unique_labels.tolist()  # Store unique labels for reference
        }
    return evaluation_results

def select_best_clustering_model(evaluation_results):
    best_model_name = None
    best_score = -1  # Start with a low score

    for name, metrics in evaluation_results.items():
        if metrics['silhouette_score'] is not None and metrics['silhouette_score'] > best_score:
            best_score = metrics['silhouette_score']
            best_model_name = name

    return best_model_name, evaluation_results[best_model_name] if best_model_name else None

trained_models = train_clustering_models(X_train_scaled)  # Use scaled data for training
evaluation_results = evaluate_clustering_models(trained_models, X_test_scaled)  # Use scaled test data
best_model_name, best_model_metrics = select_best_clustering_model(evaluation_results)
print(f"Best Model: {best_model_name}, Metrics: {best_model_metrics}")