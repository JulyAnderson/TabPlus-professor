import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from data_loading_process import load_and_preprocess_data
from model_training import train_model

# Configuração do layout do dashboard
st.set_page_config(layout="wide")
st.title('Análise dos Dados do Jogo Tab+')

# Carregar dados
df_inicial, df, multiplications_df, X_train, X_test, y_train, y_test = load_and_preprocess_data()
models, class_weights = train_model(X_train, y_train)

# Navbar lateral para selecionar a seção
st.sidebar.title("Navegação")
options = st.sidebar.radio("Selecione uma seção:", ("Visão Geral", "Análise de Turmas", "Análise Individual", "Avaliação dos Modelos"))

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

#Seção: Análise Individual 
elif options == "Análise Individual":
    st.header("Análise Individual")
    selected_player = st.selectbox("Selecione o aluno", df['player'].unique())
    player_df = df[df['player'] == selected_player]

    st.subheader(f"Evolução do Aluno {selected_player} ao Longo do Tempo")
    fig = px.line(player_df, x='game_id', y='hits', title=f"Evolução dos Acertos para {selected_player}", 
                  labels={'game_id': 'ID do Jogo', 'hits': 'Acertos'})
    st.plotly_chart(fig)


# Seção: Avaliação dos Modelos de Machine Learning
elif options == "Avaliação dos Modelos":
    from sklearn.cluster import KMeans, DBSCAN
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score

    # Seleciona colunas para agrupamento (exemplo: acertos e erros em multiplicações específicas)
    grouping_data = df[['hits', 'fator1', 'fator2']]

    # Padroniza os dados
    scaler = StandardScaler()
    grouping_data_scaled = scaler.fit_transform(grouping_data)

    # Aplica o modelo K-Means
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Cluster_KMeans'] = kmeans.fit_predict(grouping_data_scaled)

    # Calcula o Silhouette Score para o K-Means
    silhouette_kmeans = silhouette_score(grouping_data_scaled, df['Cluster_KMeans'])

    # Adiciona estatísticas por grupo para o K-Means
    group_stats_kmeans = df.groupby('Cluster_KMeans').agg({
        'hits': ['mean', 'std'],
        'multiplication': lambda x: x.value_counts().idxmax()  # Operação com maior número de erros
    }).reset_index()
    group_stats_kmeans.columns = ['Cluster', 'Mean Hits', 'Hits Std Dev', 'Most Common Multiplication Error']

    # Exibe as estatísticas dos clusters do K-Means no Streamlit
    st.header("Agrupamento de Alunos com K-Means")
    st.write(f"Silhouette Score para K-Means: {silhouette_kmeans:.2f}")
    st.write("Grupos de alunos com dificuldades em comum:")
    st.dataframe(group_stats_kmeans)

    # Exibe um gráfico de dispersão para visualização dos clusters do K-Means
    fig_kmeans = px.scatter(df, x='hits', y='multiplication', color='Cluster_KMeans', 
                            title="Distribuição dos Alunos pelos Clusters - K-Means",
                            labels={'hits': 'Acertos', 'multiplication': 'Multiplicação'})
    st.plotly_chart(fig_kmeans)

    # Aplica o modelo DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['Cluster_DBSCAN'] = dbscan.fit_predict(grouping_data_scaled)

    # Calcula o Silhouette Score para o DBSCAN (excluindo outliers rotulados como -1)
    silhouette_dbscan = silhouette_score(grouping_data_scaled[df['Cluster_DBSCAN'] != -1], 
                                         df[df['Cluster_DBSCAN'] != -1]['Cluster_DBSCAN'])

    # Adiciona estatísticas por grupo para o DBSCAN (excluindo outliers)
    group_stats_dbscan = df[df['Cluster_DBSCAN'] != -1].groupby('Cluster_DBSCAN').agg({
        'hits': ['mean', 'std'],
        'multiplication': lambda x: x.value_counts().idxmax()
    }).reset_index()
    group_stats_dbscan.columns = ['Cluster', 'Mean Hits', 'Hits Std Dev', 'Most Common Multiplication Error']

    # Exibe as estatísticas dos clusters do DBSCAN no Streamlit
    st.header("Agrupamento de Alunos com DBSCAN")
    st.write(f"Silhouette Score para DBSCAN (excluindo outliers): {silhouette_dbscan:.2f}")
    st.write("Grupos de alunos com dificuldades em comum (DBSCAN):")
    st.dataframe(group_stats_dbscan)

    # Exibe um gráfico de dispersão para visualização dos clusters do DBSCAN
    fig_dbscan = px.scatter(df[df['Cluster_DBSCAN'] != -1], x='hits', y='multiplication', color='Cluster_DBSCAN', 
                            title="Distribuição dos Alunos pelos Clusters - DBSCAN",
                            labels={'hits': 'Acertos', 'multiplication': 'Multiplicação'})
    st.plotly_chart(fig_dbscan)

    # Exibe mensagem se houverem muitos outliers detectados pelo DBSCAN
    num_outliers = len(df[df['Cluster_DBSCAN'] == -1])
    if num_outliers > 0:
        st.write(f"DBSCAN detectou {num_outliers} alunos como outliers.")

    # Conclusão sobre o modelo de agrupamento
    st.write("**Escolha do Modelo:**")
    if silhouette_kmeans > silhouette_dbscan:
        st.write("K-Means apresenta um melhor Silhouette Score e pode ser mais adequado para o agrupamento.")
    else:
        st.write("DBSCAN apresenta um Silhouette Score competitivo e pode ser melhor para detectar grupos com diferentes formas.")

st.dataframe(df)