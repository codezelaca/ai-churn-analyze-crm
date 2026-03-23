FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY src /app/src
COPY data /app/data
COPY models /app/models
COPY reports /app/reports

ENV PYTHONPATH=/app/src

EXPOSE 8081
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
