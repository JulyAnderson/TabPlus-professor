# Importações
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
import joblib
from data_loading_process import load_and_preprocess_data  # Certifique-se de que a função está corretamente importada
import os
import numpy as np

# Carregar e pré-processar os dados
df, X_train, X_test, y_train, y_test = load_and_preprocess_data('local')

def load_and_preprocess_data_for_clustering(df):
    """
    Prepara os dados para clustering garantindo normalização adequada
    """
    # Selecionar features relevantes
    features = ['hits', 'fator1', 'fator2', 'erro', 'game_grade_encoded']
    X = df[features].copy()
    
    # Normalizar os dados antes do encoding
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features

def load_models():
    """
    Carrega os modelos salvos
    """
    try:
        # Verificar se o diretório e os arquivos existem
        if not os.path.exists('models'):
            raise FileNotFoundError("Directory 'models' not found")
            
        # Carregar os modelos
        encoder_model = load_model('models/encoder_model')
        kmeans = joblib.load('models/kmeans_model.joblib')
        scaler = joblib.load('models/scaler_model.joblib')
        
        return encoder_model, kmeans, scaler
    
    except Exception as e:
        st.error(f"Erro ao carregar os modelos: {str(e)}")
        return None, None, None

encoder_model, kmeans, scaler = load_models()

# Configuração do dashboard do Streamlit
st.set_page_config(layout="wide")
st.title('Análise dos Dados do Jogo Tab+')

# Navegação na barra lateral
st.sidebar.title("Navegação")
options = st.sidebar.radio("Selecione uma seção:", 
                            ("Visão Geral", "Análise de Turmas", "Análise Individual", "Avaliação dos Modelos"))

# Seção: Visão Geral
if options == "Visão Geral":
    st.header("Visão Geral")
    col1, col2 = st.columns(2)

    with col1:
        category_count = df['multiplication'].value_counts().reset_index()
        category_count.columns = ['multiplication', 'Contagem']
        category_count = category_count[category_count['Contagem'] > 7]
        fig = px.bar(category_count, x='Contagem', 
                     y='multiplication', 
                     title="Principais Erros de Multiplicação",
                     labels={'Contagem': 'Contagem dos erros',  # nome do eixo x
                    'multiplication': 'Multiplicação'})  # nome do eixo y)
        st.plotly_chart(fig)

    with col2:
        # Criando um dicionário para mapear os códigos para os nomes completos das turmas
        turmas_map = {
            '6A': '6º ano A', '6B': '6º ano B', '6C': '6º ano C', '6D': '6º ano D',
            '7A': '7º ano A', '7B': '7º ano B', '7C': '7º ano C', '7D': '7º ano D',
            '8A': '8º ano A', '8B': '8º ano B', '8C': '8º ano C', '8D': '8º ano D',
            '9A': '9º ano A', '9B': '9º ano B', '9C': '9º ano C', '9D': '9º ano D'
        }

        performance_by_grade = df.groupby('game_grade')['hits'].mean().reset_index()
        performance_by_grade = performance_by_grade[performance_by_grade['hits'] > 0].sort_values(by='game_grade')

        # Aplicando o mapeamento na coluna game_grade
        performance_by_grade['game_grade'] = performance_by_grade['game_grade'].map(turmas_map)

        fig = px.bar(performance_by_grade, 
                    x='game_grade', 
                    y='hits', 
                    title="Desempenho Médio por Turma",
                    labels={'game_grade': 'Turma',
                            'hits': 'Pontuação'})

        st.plotly_chart(fig)

# Seção: Análise de Turmas
elif options == "Análise de Turmas":    
    # Função para criar nome formatado da turma
    def formatar_nome_turma(turma):
        ano = turma[0]
        letra = turma[1]
        return f"{ano}º ano {letra}"
    
    # Obter todas as turmas únicas
    turmas_unicas = df['game_grade'].unique()
    # Criar lista de turmas formatadas
    turmas_formatadas = [formatar_nome_turma(turma) for turma in turmas_unicas]
    
    selected_turma_formatted = st.selectbox("Selecione a turma", sorted(turmas_formatadas))
    
    # Extrair o código original da turma do nome formatado
    ano = selected_turma_formatted[0]
    letra = selected_turma_formatted[-1]
    selected_turma = f"{ano}{letra}"
    
    turma_df = df[df['game_grade'] == selected_turma]

    st.subheader(f"Distribuição dos Acertos dos Alunos na Turma {selected_turma_formatted}")
    
    # Criar gráfico de box plot usando plotly
    fig = px.box(turma_df, 
                 x='hits', 
                 y='player',
                 title=f"Distribuição de Acertos por Aluno - {selected_turma_formatted}",
                 labels={'player': 'Aluno',
                        'hits': 'Acertos'})
    
    # Personalizar o layout
    fig.update_layout(
        showlegend=False,
        xaxis_title="Acertos",
        yaxis_title="Alunos",
        height=500,
        xaxis={'tickangle': 0}  # Rotacionar labels do eixo x para melhor legibilidade
    )
    
    # Adicionar linha média
    media_turma = turma_df['hits'].mean()
    fig.add_vline(x=media_turma, 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Média da turma: {media_turma:.2f}",
                  annotation_position="top")

    st.plotly_chart(fig)
    
    # Adicionar estatísticas descritivas
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Média de Acertos", f"{turma_df['hits'].mean():.2f}")
    with col2:
        st.metric("Mediana de Acertos", f"{turma_df['hits'].median():.2f}")
    with col3:
        st.metric("Desvio Padrão", f"{turma_df['hits'].std():.2f}")

# Seção: Análise Individual
elif options == "Análise Individual":
    selected_player = st.selectbox("Selecione o aluno", df['player'].unique())
    player_df = df[df['player'] == selected_player]

    # Criar colunas para exibir gráficos lado a lado
    col1, col2 = st.columns(2)

    with col1:
        # Gráfico de linha
        fig = px.line(
            player_df, 
            x='game_id', 
            y='hits',  
            title=f"Evolução do Aluno {selected_player} ao Longo do Tempo",
            labels={'game_id': '', 'hits': 'Acertos'}
        )
        
        # Tornando o gráfico interativo
        fig.update_layout(
            xaxis=dict(
                showticklabels=False  # Remove os ticks do eixo x
            ),
            yaxis=dict(title="Acertos"),
            hovermode="x unified"  # Mostra informações completas ao passar o mouse
        )
        fig.update_traces(mode="lines+markers")  # Adiciona marcadores aos pontos da linha

        st.plotly_chart(fig)

    with col2:
        # Adicionando um gráfico de box plot para distribuição dos acertos
        fig_box = px.box(player_df, x='hits', title=f"Distribuição de Acertos de {selected_player}")
        
        # Calcular a média de acertos da turma
        turma_media = df['hits'].mean()
        
        # Adicionar linha média da turma no box plot
        fig_box.add_vline(x=turma_media, line_dash="dash", line_color="red", 
                          annotation_text="Média da Turma", 
                          annotation_position="top left")

        st.plotly_chart(fig_box)

    # Estatísticas descritivas
    st.subheader("Estatísticas Descritivas")
    mean_hits = player_df['hits'].mean()
    median_hits = player_df['hits'].median()
    std_hits = player_df['hits'].std()
    min_hits = player_df['hits'].min()
    max_hits = player_df['hits'].max()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Média de Acertos", f"{mean_hits:.2f}")
    with col2:
        st.metric("Mediana de Acertos", f"{median_hits:.2f}")
    with col3:
        st.metric("Desvio Padrão", f"{std_hits:.2f}")
    with col4:
        st.metric("Mínimo de Acertos", f"{min_hits:.2f}")
    with col5:
        st.metric("Máximo de Acertos", f"{max_hits:.2f}")

    # Comparação com a média da turma
    st.markdown(f"**Média de Acertos da Turma**: {turma_media:.2f}")

elif options == "Avaliação dos Modelos":
    if encoder_model is None or kmeans is None:
        st.error("Modelos não carregados corretamente")
    else:
        # Preparar dados
        X_scaled, features = load_and_preprocess_data_for_clustering(df)
        
        # Verificar dimensões
        st.write(f"Dimensões dos dados de entrada: {X_scaled.shape}")
        
        # Codificar dados com autoencoder
        encoded_data = encoder_model.predict(X_scaled)
        st.write(f"Dimensões dos dados codificados: {encoded_data.shape}")
        
        # Aplicar K-means
        kmeans_labels = kmeans.predict(encoded_data)
        n_clusters = len(np.unique(kmeans_labels))
        st.write(f"Número de clusters detectados: {n_clusters}")
        
        # Criar DataFrame para visualização
        plot_df = pd.DataFrame(encoded_data, columns=['Componente1', 'Componente2'])
        plot_df['Cluster'] = kmeans_labels.astype(str)  # Converter para string
        
        # Criar gráfico de dispersão
        fig = px.scatter(
            plot_df,
            x='Componente1',
            y='Componente2',
            color='Cluster',
            title="Distribuição dos Clusters",
            labels={
                'Componente1': 'Primeira Componente',
                'Componente2': 'Segunda Componente',
                'Cluster': 'Cluster'
            },
            color_discrete_sequence=px.colors.qualitative.Set3  # Usar uma paleta de cores diferente
        )
        
        # Personalizar layout
        fig.update_layout(
            # plot_bgcolor='white',
            width=800,
            height=600,
            legend=dict(
                title="Clusters",
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Adicionar grades
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        
        st.plotly_chart(fig)
        
          # Análise amigável dos clusters
        st.subheader("Análise dos Grupos de Alunos")
        
        # Criar DataFrame com dados originais e labels
        analysis_df = pd.DataFrame(X_scaled, columns=features)
        analysis_df['Cluster'] = kmeans_labels
        
        # Calcular estatísticas por cluster
        cluster_stats = analysis_df.groupby('Cluster').agg(['mean', 'std'])
        
        # Função para classificar o nível baseado na média
        def classificar_nivel(valor, metrica):
            if metrica in ['hits']:
                if valor > 0.5:
                    return "Alto ⭐"
                elif valor > 0:
                    return "Médio 📊"
                else:
                    return "Em desenvolvimento 📈"
            elif metrica in ['erro']:
                if valor < -0.5:
                    return "Baixo ✅"
                elif valor < 0:
                    return "Médio 📊"
                else:
                    return "Precisa de atenção ⚠️"
            else:
                if valor > 0.5:
                    return "Alto 📈"
                elif valor > -0.5:
                    return "Médio 📊"
                else:
                    return "Baixo 📉"
        
        # Exibir análise amigável por cluster
        for cluster in range(n_clusters):
            with st.expander(f"📊 Grupo {cluster + 1} de Alunos"):
                stats = cluster_stats.xs(cluster)
                
                # Desempenho geral
                st.markdown("### 🎯 Desempenho Geral")
                nivel_acertos = classificar_nivel(stats['hits']['mean'], 'hits')
                st.markdown(f"**Nível de Acertos**: {nivel_acertos}")
                
                # Dificuldade das atividades
                st.markdown("### 📚 Nível das Atividades")
                nivel_fator1 = classificar_nivel(stats['fator1']['mean'], 'fator1')
                nivel_fator2 = classificar_nivel(stats['fator2']['mean'], 'fator2')
                st.markdown(f"""
                - **Primeiro número**: {nivel_fator1}
                - **Segundo número**: {nivel_fator2}
                """)
                
                # Taxa de erros
                st.markdown("### ⚠️ Padrão de Erros")
                nivel_erro = classificar_nivel(stats['erro']['mean'], 'erro')
                st.markdown(f"**Taxa de erros**: {nivel_erro}")
                
                # Nível da turma
                st.markdown("### 👥 Características da Turma")
                nivel_turma = classificar_nivel(stats['game_grade_encoded']['mean'], 'game_grade')
                st.markdown(f"**Nível médio da turma**: {nivel_turma}")
                
                # Recomendações personalizadas
                st.markdown("### 💡 Recomendações")
                
                if nivel_acertos == "Em desenvolvimento 📈":
                    st.markdown("""
                    - ✨ Focar em exercícios básicos para fortalecer a base
                    - 🎯 Estabelecer metas graduais de melhoria
                    - 👥 Considerar atividades em grupo para aprendizado colaborativo
                    """)
                elif nivel_acertos == "Médio 📊":
                    st.markdown("""
                    - 📚 Aumentar gradualmente a complexidade dos exercícios
                    - 🔄 Manter prática regular
                    - 🎯 Focar em áreas específicas para melhoria
                    """)
                else:
                    st.markdown("""
                    - 🌟 Propor desafios mais avançados
                    - 👥 Incentivar mentoria entre colegas
                    - 🎯 Explorar conceitos mais complexos
                    """)
                
                # Distribuição dos alunos
                st.markdown("### 📊 Informações Adicionais")
                n_alunos = (kmeans_labels == cluster).sum()
                st.markdown(f"**Número de alunos neste grupo**: {n_alunos}")
                
                # Separador visual
                st.markdown("---")
                # Criar DataFrame com informações dos alunos e seus clusters
            alunos_cluster_df = pd.DataFrame({
                'player': df['player'],
                'Cluster': kmeans_labels
            })
            
            # Verificar duplicatas
            alunos_duplicados = alunos_cluster_df['player'].value_counts()
            alunos_em_multiplos_clusters = alunos_duplicados[alunos_duplicados > 1]
            
        # Exibir alunos em cada cluster
        st.markdown("## 👥 Distribuição dos Alunos por Grupo")
        
        if len(alunos_em_multiplos_clusters) > 0:
            st.warning(f"""
            ⚠️ **Atenção**: Foram encontrados {len(alunos_em_multiplos_clusters)} alunos em múltiplos grupos.
            Isso pode indicar que esses alunos apresentam características de diferentes perfis de aprendizado.
            """)
                
            
        # Exibir alunos por cluster
        for cluster in range(n_clusters):
            with st.expander(f"📋 Lista de Alunos - Grupo {cluster + 1}"):
                # Obter alunos do cluster atual
                alunos_no_cluster = alunos_cluster_df[alunos_cluster_df['Cluster'] == cluster]['player'].unique()
                
                if len(alunos_no_cluster) > 0:
                    # Ordenar alunos para melhor visualização
                    alunos_ordenados = sorted(alunos_no_cluster)
                    
                    # Criar colunas para exibir os alunos
                    cols = st.columns(3)
                    alunos_por_coluna = len(alunos_ordenados) // 3 + 1
                    
                    for i, col in enumerate(cols):
                        with col:
                            inicio = i * alunos_por_coluna
                            fim = min((i + 1) * alunos_por_coluna, len(alunos_ordenados))
                            alunos_coluna = alunos_ordenados[inicio:fim]
                            
                            for aluno in alunos_coluna:
                                # Verificar se o aluno está em múltiplos clusters
                                if aluno in alunos_em_multiplos_clusters:
                                    st.markdown(f"- {aluno} ⚠️")
                                else:
                                    st.markdown(f"- {aluno}")
                    
                    # Mostrar estatísticas do grupo
                    st.markdown("---")
                    st.markdown(f"""
                    📊 **Estatísticas do Grupo {cluster + 1}**:
                    - Total de alunos: {len(alunos_no_cluster)}
                    - Alunos em múltiplos grupos: {sum(1 for aluno in alunos_no_cluster if aluno in alunos_em_multiplos_clusters)}
                    """)
                else:
                    st.info("Nenhum aluno neste grupo.")

        # Adicionar métricas gerais
        st.markdown("## 📊 Métricas Gerais")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Total de Alunos",
                len(alunos_cluster_df['player'].unique())
            )
        
        with col2:
            st.metric(
                "Número de Grupos",
                n_clusters
            )
        
        with col3:
            st.metric(
                "Alunos em Múltiplos Grupos",
                len(alunos_em_multiplos_clusters)
            )


# Finalizando o aplicativo
st.sidebar.info("Use o menu à esquerda para navegar entre as seções.")
