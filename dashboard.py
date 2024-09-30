# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('games.csv')

# Set up the Streamlit app
def main():
    st.title('Dashboard de Desempenho dos Alunos')
    st.sidebar.title('Opções')

    # Análise de Desempenho Médio dos Alunos
    st.header('Desempenho Médio dos Alunos')
    student_performance = df.groupby('player')['hits'].mean().sort_values(ascending=False)
    st.bar_chart(student_performance)

    # Identificar alunos que podem precisar de mais ajuda
    st.header('Alunos que Podem Precisar de Mais Ajuda')
    students_needing_help = student_performance[student_performance < student_performance.mean()]
    st.write(students_needing_help)

    # Progresso dos Alunos ao Longo do Tempo
    st.header('Progresso dos Alunos ao Longo do Tempo')
    df['game_id'] = pd.to_numeric(df['game_id'])
    df = df.sort_values('game_id')
    df['cumulative_avg'] = df.groupby('player')['hits'].expanding().mean().reset_index(level=0, drop=True)
    for player in df['player'].unique():
        player_data = df[df['player'] == player]
        plt.plot(player_data['game_id'], player_data['cumulative_avg'], label=player)
    plt.title('Progresso dos Alunos ao Longo do Tempo')
    plt.xlabel('ID do Jogo')
    plt.ylabel('Média Cumulativa de Acertos')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    st.pyplot(plt)

    # Frequência das Operações de Multiplicação
    st.header('Frequência das Operações de Multiplicação')
    operation_frequency = df['multiplication'].value_counts()
    st.bar_chart(operation_frequency)

if __name__ == '__main__':
    main()