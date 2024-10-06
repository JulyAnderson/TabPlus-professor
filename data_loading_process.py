import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from itertools import product


#Perguntas a serem respondidas:
#1. Quais as operações que os alunos mais possuem dificuldade?
#2. Qual a melhor sala, tem como comparar?
#3. Houve evolução com o numero de partida?
#4. Existe alguma forma de agrupar esses alunos para dispensar atenção adequada a cada grupo?


def load_and_preprocess_data():
    # Load the data
    df_inicial = pd.read_csv('games_inicial.csv')  # Connect directly to the API later

    # Data Preprocessing
    # 1. Separate 'multiplication' into 'fator1' and 'fator2'
    df_inicial[['fator1', 'fator2']] = df_inicial['multiplication'].str.split('x', expand=True).astype(int)

    # 2. Create 'erro' column based on answer comparison
    df_inicial['erro'] = (df_inicial['answer'] != df_inicial['result']).astype(int)

    # 3. Generate all possible multiplications
    all_multiplications = [(i, j) for i, j in product(range(1, 16), repeat=2)]
    multiplications_df = pd.DataFrame(all_multiplications, columns=['fator1', 'fator2'])


    # 4. Encode categorical variables
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

    # Visualizar as primeiras linhas para verificar a anonimização
    df[['player', 'player_encoded', 'player_anonymized']].head()

    # Substituir a coluna 'player' pelos valores anonimizados da coluna 'player_anonymized'
    df['player'] = df['player_anonymized']

    # Remover a coluna temporária 'player_anonymized'
    df = df.drop(columns=['player_anonymized'])


    # 5. Define features and target
    features = df[['fator1', 'fator2', 'hits', 'game_grade_encoded', 'game_year']]
    target = df['erro']

    # 6. Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    return df_inicial, df, multiplications_df ,  X_train, X_test, y_train, y_test

