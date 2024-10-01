# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
<<<<<<< HEAD
 
st.set_page_config(layout='wide')
=======
from sklearn.preprocessing import StandardScaler
>>>>>>> bfaa4958a25f1a607a4dff95f80ce0ad708716ee

# Load the data
df = pd.read_csv('games.csv')

<<<<<<< HEAD

col1, col2, col3 = st.columns(3)
col4, col5, col6 = st.columns(3)

# Adiciona conteúdo à primeira coluna
with col1:
    st.header("Resumo Estatístico")
    st.dataframe(df['hits'].describe())

# Adiciona conteúdo à segunda coluna
with col2:
    st.header("Coluna 2")
    st.write("Esta é a segunda coluna.")

# Adiciona conteúdo à terceira coluna
with col3:
    st.header("Coluna 3")
    st.write("Esta é a terceira coluna.")
=======
# 2. Exibir o DataFrame no Streamlit
st.title('Análise do Jogo de Multiplicação')
st.write('Dados do Jogo:')
st.dataframe(df)

# 3. Pré-processamento
features = df[['hits', 'fator1', 'fator2', 'result']]

# Normalizando os dados

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# 4. Aplicação do K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster_kmeans'] = kmeans.fit_predict(scaled_features)

# 5. Gráfico de Dispersão interativo com clusters
fig = px.scatter(df, x='fator1', y='fator2', color='cluster_kmeans', 
                 title='Clusters de Jogadores com base nos Fatores')
st.plotly_chart(fig)

# 6. Gráficos adicionais
st.write('Distribuição de Resultados por Cluster:')
fig2 = px.histogram(df, x='result', color='cluster_kmeans', nbins=10, title='Histograma de Resultados')
st.plotly_chart(fig2)
>>>>>>> bfaa4958a25f1a607a4dff95f80ce0ad708716ee
