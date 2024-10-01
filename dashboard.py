# Import necessary libraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
st.set_page_config(layout='wide')

# Load the data
df = pd.read_csv('games.csv')


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
