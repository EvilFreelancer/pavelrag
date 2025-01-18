FROM python:3.12
WORKDIR /app
EXPOSE 5000

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY . .

ENTRYPOINT ["/app/entrypoint.sh"]
