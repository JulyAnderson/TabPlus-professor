import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import product

#Perguntas a serem respondidas:
#1. Quais as operações que os alunos mais possuem dificuldade?
#2. Qual a melhor sala, tem como comparar?
#3. Houve evolução com o numero de partida?
#4. Existe alguma forma de agrupar esses alunos para dispensar atenção adequada a cada grupo?

import pandas as pd
import requests
import json
from itertools import product
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def get_all_games():
    base_url = "https://tabplusbackend.onrender.com/Game"
    endpoint = "/searchAll"
    url = base_url + endpoint

    try:
        response = requests.get(url)
        # Verifica se a resposta foi bem-sucedida (código 200)
        if response.status_code == 200:
            # Retorna os dados em formato JSON
            games = response.json()
            return games
        else:
            print("Erro na requisição:", response.status_code)
            return None
    except Exception as e:
        print("Erro na requisição:", e)
        return None

def load_and_preprocess_data(source='api'):
    """
    Carrega e pré-processa os dados.
    
    :param source: Fonte dos dados. Pode ser 'api' ou 'local'. Padrão é 'api'.
    :return: DataFrame processado, X_train, X_test, y_train, y_test
    """
    if source not in ['api', 'local']:
        raise ValueError("A fonte deve ser 'api' ou 'local'")

    if source == 'api':
        # Tentar buscar os dados da API
        games = get_all_games()
        
        if games is None:
            print("Falha ao carregar dados da API. Usando dados locais.")
            df_inicial = pd.read_csv('games_inicial.csv')
        else:
            print("Dados carregados da API.")
            df_inicial = pd.DataFrame(games)
            df_inicial.to_csv('games_inicial.csv', index=False)  # Salvar os dados recebidos no CSV
    else:
        print("Carregando dados do arquivo local 'games_inicial.csv'.")
        df_inicial = pd.read_csv('games_inicial.csv')

    # Data Preprocessing
    # 1. Separar 'multiplication' em 'fator1' e 'fator2'
    df_inicial[['fator1', 'fator2']] = df_inicial['multiplication'].str.split('x', expand=True).astype(int)

    # 2. Criar coluna 'erro' baseada na comparação de respostas
    df_inicial['erro'] = (df_inicial['answer'] != df_inicial['result']).astype(int)

    # 3. Gerar todas as multiplicações possíveis
    all_multiplications = [(i, j) for i, j in product(range(1, 16), repeat=2)]
    multiplications_df = pd.DataFrame(all_multiplications, columns=['fator1', 'fator2'])

    # 4. Codificar variáveis categóricas
    df = df_inicial.copy()
    label_encoder_grade = LabelEncoder()
    label_encoder_player = LabelEncoder()
    df['game_grade_encoded'] = label_encoder_grade.fit_transform(df['game_grade'])
    df['player_encoded'] = label_encoder_player.fit_transform(df['player'])

    # Criar um mapeamento para substituir valores únicos de 'player_encoded' por rótulos genéricos
    unique_players = df['player_encoded'].unique()
    player_mapping = {player: f'Jogador {i+1}' for i, player in enumerate(unique_players)}

    # Aplicar o mapeamento para anonimizar a coluna 'player_encoded'
    df['player_anonymized'] = df['player_encoded'].map(player_mapping)

    # Substituir a coluna 'player' pelos valores anonimizados da coluna 'player_anonymized'
    df['player'] = df['player_anonymized']

    # Remover a coluna temporária 'player_anonymized'
    df = df.drop(columns=['player_anonymized'])

    # 5. Definir características e alvo
    features = df[['fator1', 'fator2', 'hits', 'game_grade_encoded', 'game_year']]
    target = df['erro']

    # 6. Dividir dados em conjuntos de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return df, X_train, X_test, y_train, y_test