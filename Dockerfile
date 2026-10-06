FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY backend ./backend
RUN pip install --no-cache-dir .
COPY data/rules ./data/rules
EXPOSE 8000
CMD ["uvicorn", "rxsentinel.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
