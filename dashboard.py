# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('games.csv')

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