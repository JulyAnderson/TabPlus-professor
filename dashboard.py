import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)

from data_loading_process import load_and_preprocess_data
from model_training import train_model

# Load data and preprocess
df_inicial, df, multiplications_df, df_identificacao_encoder, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Train models
models, class_weights = train_model(X_train, y_train)

# Title of the dashboard
st.title('Dashboard de Análise de Dificuldades dos Alunos')

# Section 1: Data Overview
st.subheader('Dados do Jogo:')
st.dataframe(df_inicial)
st.dataframe(df)
st.dataframe(multiplications_df)
st.dataframe(df_identificacao_encoder)

# Section 2: K-Means Clustering
st.header('Análise de Clusters de Jogadores')
# Add your K-Means clustering code here...

# Section 3: Select Class for Analysis
# Add your class selection code here...

# Section 4: Model Training and Evaluation
st.title("Modelo de Machine Learning para Análise de Desempenho")

# Evaluate models
for name, model in models.items():
    y_pred = model.predict(X_test)

    st.write(f"**Model:** {name}")
    st.write(f"**Acurácia:** {accuracy_score(y_test, y_pred):.2f}")

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    st.write("**Relatório de Classificação:**")
    st.write(pd.DataFrame(report).transpose())

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Erro', 'Acertou'], yticklabels=['Erro', 'Acertou'])
    plt.ylabel('Real')
    plt.xlabel('Previsto')
    plt.title('Matriz de Confusão')
    st.pyplot(plt)
    plt.clf()  # Clear the figure for the next plot

    # ROC Curve
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    st.pyplot(plt)
    plt.clf()  # Clear the figure for the next plot

    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(10, 6))
        plt.title('Importância das Características')
        plt.bar(range(X_train.shape[1]), importances[indices], align='center')
        plt.xticks(range(X_train.shape[1]), [X_train.columns[i] for i in indices], rotation=90)
        plt.xlim([-1, X_train.shape[1]])
        st.pyplot(plt)
        plt.clf()  # Clear the figure for the next plot
