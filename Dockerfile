FROM python:3.9-slim
# Crear carpeta donde trabajar
WORKDIR /app 

# Instalar dependencias
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar el requirements
COPY requirements.txt .

# Instalar las librerías de Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto de archivos
COPY app.py .
COPY churn_pipeline.pkl .

# Exponer puerto que usa Streamlit
EXPOSE 8501

# Comando para ejecutar la aplicación al iniciar el contenedor
CMD ["sh", "-c", "streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]