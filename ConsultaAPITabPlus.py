#!/usr/bin/env python
# coding: utf-8

# # Tab+: Analisando dados dos jogos.

# O Tab+ é um jogo de navegador do tipo endless runner (corrida infinita) direcionado as alunos do Ensino Fundamental II para incentivar a aprendizagem da tabuada.\
# O jogo coleta informações e persiste dados durante a partida e abaixo apresentamos um dicionário desses dados.

# ## Dicionário de dados

# **game_id**: é o identificador único da partida do tipo Inteiro;\
# **game_grade**: é uma string na forma AnoTurma que identifica as diferentes turmas dos jogadores;\
# **game_year**: é um número inteiro que representa em qual ano a partida foi jogada;\
# **player**: é um identificador do jogador na partida do tipo string;\
# **hits**: é um número inteiro que indica a pontuação do Jogador;\
# **multiplication**: é uma string que mostra qual a multiplicação que encerrou o jogo, ou seja, qual a multiplicação o jogador não conseguiu acertar;\
# **answer**:	é um número inteiro que indica qual a resposta incorreta informada pelo jogador;\
# **result**: é um número inteiro que indica a resposta correta para a questão que gerou o encerramento da partida;
# 
# 

# ## Importando as bibliotecas

# In[ ]:


import requests
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# # Estabelecendo conexão com a API

# In[2]:


# def get_all_games():
#     base_url = "https://tabplusbackend.onrender.com/Game"
#     endpoint = "/searchAll"
#     url = base_url + endpoint

#     try:
#         response = requests.get(url)
#         # Verifica se a resposta foi bem-sucedida (código 200)
#         if response.status_code == 200:
#             # Retorna os dados em formato JSON
#             games = response.text
#             games = json.loads(games)
#             return games
#         else:
#             print("Erro na requisição:", response.status_code)
#             return None
#     except Exception as e:
#         print("Erro na requisição:", e)
#         return None

# games = get_all_games()

# df = pd.DataFrame(games)
# df.to_csv('games_inicial.csv', index=False)
# df.head(2)


# ## Carregando o DataFrame inicial

# In[2]:


df = pd.read_csv('games_inicial.csv')


# Encontrando os outliers usados para testes, através da pontuação.

<<<<<<< HEAD
=======
# In[ ]:
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


df.sort_values(by= "hits", ascending = False)


# Eliminando os valores maiores de 35 pontos, pois esses valores são de testes e podem influenciar a análise posterior.

<<<<<<< HEAD
=======
# In[4]:
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


df = df.drop(df[df['hits'] >  100 ].index)


# Buscando os valores únicos para a variável player

<<<<<<< HEAD
=======
# In[ ]:
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


jogadores_unicos= df.player.unique()
# jogadores_unicos = list(jogadores_unicos)
# jogadores_unicos
jogadores_unicos


# Deletando jogadores de teste.

<<<<<<< HEAD
=======
# In[6]:
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


players_para_eliminar = ['Admin', 'and', 'Teste']
df = df[~df['player'].isin(players_para_eliminar)]
df.to_csv('games.csv', index=False)


<<<<<<< HEAD
=======
# In[ ]:
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


# 1. Contar a ocorrência de cada jogador
contagem = df['player'].value_counts()

# 2. Criar a máscara booleana para valores que aparecem exatamente uma vez
mascara = contagem[contagem == 1].index

# 3. Filtrar o DataFrame original com a máscara booleana
df_filtrado = df[df['player'].isin(mascara)]


# Exibir o DataFrame filtrado
df_filtrado


# In[ ]:


# Create a mapping of unique player names to generic labels
unique_players = df['player'].unique()
player_mapping = {player: f'Jogador {i + 1}' for i, player in enumerate(unique_players)}

# Replace player names in the DataFrame using the mapping
df['player'] = df['player'].map(player_mapping)

<<<<<<< HEAD
=======
# Display the updated DataFrame
st.dataframe(df)

>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2

# Os dados serão anonimizados eliminando a identificação do jogador.

# In[9]:


df = df[['game_id', 'hits','multiplication', 'answer', 'result']]
df.to_csv('games_anonimo.csv', index=False)


# Verificando os dados e seus tipos. Observamos também se existe registros nulos.

# In[ ]:


df.info()


# Verificando algumas caracteristas estatísticas.

# #Descrição Estatística

# In[ ]:


df.describe()


# Nestes dados podemos observar que a maior pontuação (hints) é de 35 pontos.\
# Ainda, a mediana, ou seja, a pontuação mais comum é a de 3 pontos.\
# 75% das partidas foram inferiores a 7 pontos.\
# 50% fizeram menos de 3 pontos.

# Vamos filtrar o DataFrame para apresentar as multiplicações em que houveram mais erros por partida (mais que dois erros)

# In[12]:


df_multiplication = pd.DataFrame (df.multiplication.value_counts())
df_multiplication_maior_que_dois=  df_multiplication[df_multiplication['count'] > 2]


# In[ ]:


# Cria o gráfico de barras horizontal
plt.figure(figsize=(10, 10))  # Define o tamanho do gráfico
plt.barh(df_multiplication_maior_que_dois.index, df_multiplication_maior_que_dois['count'])  # Plota o gráfico de barras horizontal
plt.title('Contagem de Erros Mais Comuns por Multiplicações')  # Adiciona o título do gráfico
plt.xlabel('Contagem de erros por multiplicação')  # Adiciona o rótulo do eixo x
plt.ylabel('Multiplicação')  # Adiciona o rótulo do eixo y
plt.gca().invert_yaxis()  # Inverte o eixo y para que as barras apareçam na ordem correta
plt.tight_layout()  # Ajusta o layout para evitar que as labels se sobreponham
plt.show()  # Exibe o gráfico


# Agora, vamos filtrar as multiplicações que houveram menos erros (2 ou menos)

# In[14]:


df_multiplication = pd.DataFrame (df.multiplication.value_counts())
df_multiplication_menor_que_dois=  df_multiplication[df_multiplication['count'] <= 2]


# In[ ]:


# Cria o gráfico de barras horizontal
plt.figure(figsize=(10, 15))  # Define o tamanho do gráfico
plt.barh(df_multiplication_menor_que_dois.index, df_multiplication_menor_que_dois['count'])  # Plota o gráfico de barras horizontal
plt.title('Contagem de Erros Menos Comuns por Multiplicações')  # Adiciona o título do gráfico
plt.xlabel('Contagem de erros por multiplicação')  # Adiciona o rótulo do eixo x
plt.ylabel('Multiplicação')  # Adiciona o rótulo do eixo y
plt.gca().invert_yaxis()  # Inverte o eixo y para que as barras apareçam na ordem correta
plt.tight_layout()  # Ajusta o layout para evitar que as labels se sobreponham
plt.show()  # Exibe o gráfico


# Vamos gerar um novo DataFrame com todas as multiplicações possíveis entre 0x0 e 15x10 e depois compará-las à nossa amostragem. Assim, vamos identificar quais as questões não foram erradas ou então, não foram apresentadas aos alunos.

# In[ ]:


# Lista para armazenar os resultados das multiplicações
multiplicacoes = []

# Loop para calcular as multiplicações de 0x0 a 15x10
for i in range(16):  # Loop para o multiplicando (0 a 15)
    for j in range(11):  # Loop para o multiplicador (0 a 10)
        multiplicacao_str = f"{i} x {j}"  # Converte para string
        multiplicacoes.append(multiplicacao_str)  # Adiciona à lista

# Cria um DataFrame a partir da lista de resultados
df_multiplicacoes = pd.DataFrame(multiplicacoes, columns=['Multiplicação'])

# Exibe o DataFrame
df_multiplicacoes


# In[ ]:


valores_sem_correspondencia = df_multiplicacoes[~df_multiplicacoes['Multiplicação'].isin(df['multiplication'])]
<<<<<<< HEAD
print (f"Multiplicações onde não houverem erros: {valores_sem_correspondencia['Multiplicação'].tolist()}")
=======
pprint (f"Multiplicações onde não houverem erros: {valores_sem_correspondencia['Multiplicação'].tolist()}")
>>>>>>> 22d28dbc861d835135b4198f8467a125b61c16b2


# Analisando a distribuição de acertos pelo boxplot. Observamos que há um aluno que se destaca bastante nos acertos (35 pontos). A mediana se confirma em 3 pontos.

# In[ ]:


# Dados para o boxplot (supondo que 'df' seja seu DataFrame)
dados = df['hits']
# Calcula a mediana
mediana = dados.median()

# Cria o boxplot com Seaborn
sns.boxplot(x=dados, notch=True, orient='h', width=0.5, color='lightblue')

plt.axvline(x=mediana, color='blue', linestyle='--', linewidth=2)

# Adiciona título ao gráfico
plt.title('Pontuação dos Jogadores')

# Adiciona rótulo aos eixos
plt.xlabel('Pontuação')

# Exibe o boxplot
plt.show()


# In[ ]:


df.head()


# In[20]:


# Separar a coluna 'multiplication' em duas colunas 'fator1' e 'fator2'
df[['fator1', 'fator2']] = df['multiplication'].str.split('x', expand=True)

# Converter as novas colunas para o tipo inteiro
df['fator1'] = df['fator1'].astype(int)
df['fator2'] = df['fator2'].astype(int)


# In[ ]:


from sklearn.cluster import KMeans

# Usar as colunas numéricas (hits, result, fator1, fator2) para o clustering
X = df[['hits', 'result', 'fator1', 'fator2']]

# Aplicar o algoritmo KMeans com 2 clusters
kmeans = KMeans(n_clusters=9)
kmeans.fit(X)

# Adicionar as previsões (clusters) ao DataFrame
df['cluster'] = kmeans.labels_

df


# In[ ]:


import matplotlib.pyplot as plt

# Plotar os clusters usando 'fator1' e 'fator2'
plt.scatter(df['fator1'], df['fator2'], c=df['cluster'], cmap='viridis', marker='o')
plt.xlabel('Fator 1')
plt.ylabel('Fator 2')
plt.title('Clusters of Multiplication Factors')
plt.colorbar(label='Cluster')  # Adiciona uma barra de cores para identificar os clusters
plt.grid(True)  # Adiciona uma grade para melhor visualização
plt.show()


# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotar os clusters em 3D
ax.scatter(df['fator1'], df['fator2'], df['result'], c=df['cluster'], cmap='viridis', marker='o')
ax.set_xlabel('Fator 1')
ax.set_ylabel('Fator 2')
ax.set_zlabel('Result')
ax.set_title('3D Clusters of Multiplication Factors and Result')

plt.show()


# In[ ]:


from sklearn.metrics import silhouette_score

# Calcular o Silhouette Score
score = silhouette_score(X, kmeans.labels_)
print(f'Silhouette Score: {score:.2f}')


# In[ ]:


inertia = kmeans.inertia_
print(f'Inertia: {inertia:.2f}')


# In[ ]:


inertias = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K, inertias, 'bx-')
plt.xlabel('Número de Clusters')
plt.ylabel('Inertia')
plt.title('Método do Cotovelo')
plt.show()


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN

# Etapa de preparação - Excluir colunas irrelevantes (você pode ajustar conforme seus dados)
df_clean = df.select_dtypes(include=['float64', 'int64']).dropna()

# Padronização dos dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clean)

# Implementando K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(df_scaled)

# Implementando DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(df_scaled)

# Adicionando os rótulos aos dados originais
df['KMeans_Cluster'] = kmeans_labels
df['DBSCAN_Cluster'] = dbscan_labels

# Visualização dos clusters
sns.scatterplot(x=df_clean.iloc[:, 0], y=df_clean.iloc[:, 1], hue=kmeans_labels, palette='viridis')
plt.title('K-Means Clustering')
plt.show()

sns.scatterplot(x=df_clean.iloc[:, 0], y=df_clean.iloc[:, 1], hue=dbscan_labels, palette='viridis')
plt.title('DBSCAN Clustering')
plt.show()


# In[ ]:


import streamlit as st
import pandas as pd
import plotly.express as px

# Carregar o dataset
df = pd.read_csv("games.csv")

# Criar um seletor de turma
turmas = df['game_grade'].unique()
turma_selecionada = st.selectbox('Selecione a turma:', turmas)

# Filtrar os dados da turma selecionada
df_turma = df[df['game_grade'] == turma_selecionada]

# Calcular métricas relevantes
acertos_por_nivel = df_turma.groupby('game_grade')['hits'].mean().reset_index()

# Gráfico de barras - Acertos por nível
fig = px.bar(acertos_por_nivel, x='game_grade', y='hits', title='Acertos Médios por Nível')
st.plotly_chart(fig)




# In[ ]:


df


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Configurar o estilo dos gráficos
sns.set(style='whitegrid')

# Analisar a distribuição de acertos
plt.figure(figsize=(10, 6))
sns.histplot(df['hits'], bins=20, kde=True)
plt.title('Distribuição de Acertos')
plt.xlabel('Número de Acertos')
plt.ylabel('Frequência')
plt.show()

# Analisar as multiplicações mais difíceis
# Criar uma coluna para verificar se a resposta está correta
df['correct'] = df['answer'] == df['result']

# Agrupar por multiplicação e calcular a taxa de acertos
multiplication_difficulty = df.groupby('multiplication')['correct'].mean().sort_values()

# Visualizar as multiplicações mais difíceis
plt.figure(figsize=(12, 8))
multiplication_difficulty.plot(kind='barh')
plt.title('Multiplicações Mais Difíceis (Menor Taxa de Acertos)')
plt.xlabel('Taxa de Acertos')
plt.ylabel('Multiplicação')
plt.show()


# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Calcular a m\u00e9dia de acertos por aluno
student_performance = df.groupby('player')['hits'].mean().sort_values(ascending=False)

# Visualizar o desempenho m\u00e9dio dos alunos
plt.figure(figsize=(12, 6))
student_performance.plot(kind='bar')
plt.title('Desempenho M\u00e9dio dos Alunos')
plt.xlabel('Aluno')
plt.ylabel('M\u00e9dia de Acertos')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Identificar alunos que podem precisar de mais ajuda
students_needing_help = student_performance[student_performance < student_performance.mean()]

print("Alunos que podem precisar de mais ajuda:")
print(students_needing_help)

# Analisar o progresso ao longo do tempo
df['game_id'] = pd.to_numeric(df['game_id'])
df = df.sort_values('game_id')
df['cumulative_avg'] = df.groupby('player')['hits'].expanding().mean().reset_index(level=0, drop=True)

# Visualizar o progresso dos alunos ao longo do tempo
plt.figure(figsize=(12, 6))
for player in df['player'].unique():
    player_data = df[df['player'] == player]
    plt.plot(player_data['game_id'], player_data['cumulative_avg'], label=player)

plt.title('Progresso dos Alunos ao Longo do Tempo')
plt.xlabel('ID do Jogo')
plt.ylabel('M\u00e9dia Cumulativa de Acertos')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# An\u00e1lise das opera\u00e7\u00f5es mais frequentes
operation_frequency = df['multiplication'].value_counts()

plt.figure(figsize=(12, 6))
operation_frequency.plot(kind='bar')
plt.title('Frequ\u00eancia das Opera\u00e7\u00f5es de Multiplica\u00e7\u00e3o')
plt.xlabel('Opera\u00e7\u00e3o')
plt.ylabel('Frequ\u00eancia')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[ ]:




