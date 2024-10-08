# Importações
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from itertools import product
from keras.models import Model, load_model
from keras.layers import Input, Dense
import joblib

def load_and_preprocess_data():
    # Carregar os dados
    df = pd.read_csv('games_inicial.csv')

    # Pré-processamento dos dados
    df[['fator1', 'fator2']] = df['multiplication'].str.split('x', expand=True).astype(int)
    df['erro'] = (df['answer'] != df['result']).astype(int)

    # Codificar variáveis categóricas
    label_encoder_grade = LabelEncoder()
    label_encoder_player = LabelEncoder()
    df['game_grade_encoded'] = label_encoder_grade.fit_transform(df['game_grade'])
    df['player_encoded'] = label_encoder_player.fit_transform(df['player'])

    # Substituir valores únicos de 'player_encoded' por rótulos genéricos
    unique_players = df['player_encoded'].unique()
    player_mapping = {player: f'Jogador {i + 1}' for i, player in enumerate(unique_players)}
    df['player'] = df['player_encoded'].map(player_mapping)

    # Definir características e alvo
    features = df[['hits', 'fator1', 'fator2','erro','game_grade_encoded']]
    target = df['erro']

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return df, X_train, X_test, y_train, y_test

def load_models():
    # Carregar o modelo de autoencoder
    encoder_model = load_model('encoder_model.h5')
    # Carregar o modelo de K-means
    kmeans = joblib.load('kmeans_model.pkl')
    return encoder_model, kmeans

# Configuração do dashboard do Streamlit
st.set_page_config(layout="wide")
st.title('Análise dos Dados do Jogo Tab+')

# Carregar dados
df, X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Navegação na barra lateral
st.sidebar.title("Navegação")
options = st.sidebar.radio("Selecione uma seção:", 
                            ("Visão Geral", "Análise de Turmas", "Análise Individual", "Avaliação dos Modelos"))

# Seção: Visão Geral
if options == "Visão Geral":
    st.header("Visão Geral")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Principais Erros de Multiplicação")
        category_count = df['multiplication'].value_counts().reset_index()
        category_count.columns = ['multiplication', 'Contagem']
        category_count = category_count[category_count['Contagem'] > 7]
        fig = px.bar(category_count, x='Contagem', y='multiplication', title="Multiplicações com Mais Erros")
        st.plotly_chart(fig)

    with col2:
        st.subheader("Comparação de Performance entre Turmas")
        performance_by_grade = df.groupby('game_grade')['hits'].mean().reset_index()
        performance_by_grade = performance_by_grade[performance_by_grade['hits'] > 0].sort_values(by='game_grade')
        fig = px.bar(performance_by_grade, x='game_grade', y='hits', title="Desempenho Médio por Turma")
        st.plotly_chart(fig)

# Seção: Análise de Turmas
elif options == "Análise de Turmas":
    st.header("Análise de Turmas")
    selected_turma = st.selectbox("Selecione a turma", df['game_grade'].unique())
    turma_df = df[df['game_grade'] == selected_turma]

    st.subheader(f"Distribuição dos Acertos dos Alunos na Turma {selected_turma}")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='player', y='hits', data=turma_df, ax=ax)
    ax.set_title("Distribuição de Acertos por Aluno")
    ax.set_xlabel("Aluno")
    ax.set_ylabel("Acertos")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Seção: Análise Individual
elif options == "Análise Individual":
    st.header("Análise Individual")
    selected_player = st.selectbox("Selecione o aluno", df['player'].unique())
    player_df = df[df['player'] == selected_player]

    st.subheader(f"Evolução do Aluno {selected_player} ao Longo do Tempo")
    fig = px.line(player_df, x='game_id', y='hits', title=f"Evolução dos Acertos para {selected_player}", 
                  labels={'game_id': 'ID do Jogo', 'hits': 'Acertos'})
    st.plotly_chart(fig)

# Seção: Avaliação dos Modelos
elif options == "Avaliação dos Modelos":
    st.header("Avaliação dos Modelos")

    # Carregar os modelos
    encoder_model, kmeans = load_models()

    # Certifique-se de que os dados de entrada estão corretos
    input_shape = encoder_model.input_shape[1]
    if X_train.shape[1] != input_shape:
        st.error(f"O modelo espera uma entrada de {input_shape} características, mas recebeu {X_train.shape[1]}.")
    else:
        # Codificar os dados de entrada usando o encoder
        encoded_data = encoder_model.predict(X_train)

        # Obter os clusters
        kmeans_labels = kmeans.predict(encoded_data)

        # Combinar os dados originais com os rótulos de cluster
        cluster_df = pd.DataFrame(X_train, columns=['hits', 'fator1', 'fator2', 'erro', 'game_grade_encoded'])
        cluster_df['Cluster'] = kmeans_labels

        # Calcular estatísticas descritivas para cada cluster
        cluster_stats = cluster_df.groupby('Cluster').agg({
            'hits': ['mean', 'std'],
            'fator1': ['mean', 'std'],
            'fator2': ['mean', 'std'],
            'erro': ['mean', 'std'],
            'game_grade_encoded': ['mean', 'std']
        })

        # Exibir as estatísticas dos clusters
        st.write("Estatísticas Descritivas dos Clusters")
        st.write(cluster_stats)

        # Gerar descrições textuais dos clusters
        st.subheader("Descrições dos Clusters")
        for cluster in cluster_stats.index:
            stats = cluster_stats.loc[cluster]
            description = f"""
            Cluster {cluster}:
            - Média de acertos: {stats[('hits', 'mean')]:.2f} (±{stats[('hits', 'std')]:.2f})
            - Fator 1 médio: {stats[('fator1', 'mean')]:.2f} (±{stats[('fator1', 'std')]:.2f})
            - Fator 2 médio: {stats[('fator2', 'mean')]:.2f} (±{stats[('fator2', 'std')]:.2f})
            - Taxa de erro média: {stats[('erro', 'mean')]:.2f} (±{stats[('erro', 'std')]:.2f})
            - Nível de turma médio: {stats[('game_grade_encoded', 'mean')]:.2f} (±{stats[('game_grade_encoded', 'std')]:.2f})
            
            Interpretação: Este cluster representa alunos com um nível de desempenho 
            {'alto' if stats[('hits', 'mean')] > cluster_stats[('hits', 'mean')].mean() else 'baixo'}, 
            trabalhando com multiplicações de dificuldade 
            {'alta' if stats[('fator1', 'mean')] * stats[('fator2', 'mean')] > (cluster_stats[('fator1', 'mean')] * cluster_stats[('fator2', 'mean')]).mean() else 'baixa'}, 
            e com uma taxa de erro 
            {'alta' if stats[('erro', 'mean')] > cluster_stats[('erro', 'mean')].mean() else 'baixa'}.
            """
            st.write(description)

        # Visualizar os clusters (mantenha a visualização original)
        st.subheader("Visualização dos Clusters com Autoencoder + K-means")
        fig = plt.figure(figsize=(8, 6))
        plt.scatter(encoded_data[:, 0], encoded_data[:, 1], c=kmeans_labels, cmap='viridis')
        plt.title('Clusters Identificados com Autoencoder e K-means')
        plt.xlabel('Feature 1 (Codificada)')
        plt.ylabel('Feature 2 (Codificada)')
        st.pyplot(fig)