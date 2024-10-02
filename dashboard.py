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

st.set_page_config(layout="wide")

# Load data and preprocess
df_inicial, df, multiplications_df, X_train, X_test, y_train, y_test = load_and_preprocess_data()


# Train models
models, class_weights = train_model(X_train, y_train)

# Title of the dashboard
st.title('Dashboard de Análise de Dificuldades dos Alunos')

col1, col2 = st.columns(2)


# Section 1: Errors in multiplications
with col1:
    # count foreach categorical
    category_count = df['multiplication'].value_counts().reset_index()
    category_count.columns = ['multiplication', 'Contagem']
    category_count = category_count[category_count['Contagem'] > 3]

    # Barchart 
    fig = px.bar(category_count, x='multiplication', y='Contagem', title ="Principais Erros na Multiplicação")

    # Show the chart no Streamlit
    st.plotly_chart(fig)

with col2:
    # Section 2: Comparing classes
    # Group by class and calculate mean of hits
    performance_by_grade = df_inicial.groupby('game_grade')['hits'].mean().reset_index()

    # Filter out classes with a mean of hits equal to 0
    performance_by_grade = performance_by_grade[performance_by_grade['hits'] > 0]

    # Sort the data by game_grade
    performance_by_grade = performance_by_grade.sort_values(by='game_grade')

    # Create comparison bar chart
    fig = px.bar(performance_by_grade, x='game_grade', y='hits', labels={'hits': 'Average Hits'}, title="Comparação de Performance entre as turmas")

    # Display in Streamlit
    st.plotly_chart(fig)


# Seletor de turma
selected_turma = st.selectbox("Selecione a turma", df_inicial['game_grade'].unique())

# Filtro da turma selecionada
turma_df = df_inicial[df_inicial['game_grade'] == selected_turma]

# Seletor para permitir a comparação do desempenho dos alunos
selected_players = st.multiselect("Selecione os alunos para comparar", turma_df['player'].unique())

if selected_players:
    st.subheader(f"Comparando o desempenho dos alunos na turma {selected_turma}")

    # Filtrar os dados para os alunos selecionados
    filtered_df = turma_df[turma_df['player'].isin(selected_players)]

    # Plotar a evolução dos acertos em um único gráfico
    plt.figure(figsize=(10, 4))
    
    for player in selected_players:
        player_data = filtered_df[filtered_df['player'] == player]
        plt.plot(player_data['game_grade'], player_data['hits'], marker='o', label=player,)
    
    plt.xlabel('Jogo (game_grade)')
    plt.ylabel('Acertos')
    plt.title(f"Evolução de Acertos na Turma {selected_turma}")
    plt.legend()
    st.pyplot(plt)


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
