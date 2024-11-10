# Use uma imagem base do Python
FROM python:3.9-slim

# Define o diretório de trabalho
WORKDIR /app

# Copia os arquivos de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copia o resto dos arquivos do projeto
COPY . .

# Cria um script de inicialização para executar os arquivos na ordem correta
RUN echo '#!/bin/bash\n\
python model_training.py && \
python model_evaluation.py && \
streamlit run app.py --server.port 8051 --server.address 0.0.0.0' > /app/start.sh

# Dá permissão de execução ao script
RUN chmod +x /app/start.sh

# Expõe a porta que o dashboard irá utilizar
EXPOSE 8051

# Define o comando para iniciar a aplicação
CMD ["/app/start.sh"]
