FROM python:3.10-slim

WORKDIR /app

# Instala dependências de build necessárias para compilar pacotes python, se necessário
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala as dependências do Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Instalação direta do modelo do spaCy a partir do lançamento do GitHub para evitar erro 404
RUN pip install https://github.com/explosion/spacy-models/releases/download/pt_core_news_sm-3.7.0/pt_core_news_sm-3.7.0.tar.gz

# Pré-download do modelo do SentenceTransformer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Pré-download do modelo do PySentimiento (Análise de Sentimentos)
RUN python -c "from pysentimiento import create_analyzer; create_analyzer(task='sentiment', lang='pt')"

# Copia o código-fonte da aplicação
COPY app.py nlp_pipeline.py .

# Expõe a porta usada pela API Flask
EXPOSE 5000

# Executa o servidor Flask utilizando o Gunicorn para produção
# -w 1: Utiliza 1 worker devido ao alto consumo de memória dos modelos de IA
# --timeout 120: Tempo limite estendido para requisições de análise mais longas
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "app:app", "--timeout", "120"]
