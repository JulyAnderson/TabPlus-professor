# Escolha uma imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copie os arquivos de requirements.txt e instale as dependências
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copie todo o conteúdo do projeto para o contêiner
COPY . .

# Exponha a porta do Streamlit
EXPOSE 8501

# Defina o comando para iniciar o Streamlit
CMD ["streamlit", "run", "app.py"]
