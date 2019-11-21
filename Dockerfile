FROM python:3.7-slim-stretch

LABEL version="1.0.0" \
    description="Serviço REST para classificação de texto de solicitações da consultoria do Senado" \
    maintainer="Senado Federal"

RUN apt-get update && apt-get install -y git python3-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade -r requirements.txt

COPY app app/

# realiza do download do modelo do classificador
RUN python app/server.py

EXPOSE 5042

CMD ["python", "app/server.py", "serve"]
