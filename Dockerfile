FROM python:3.12-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps --no-build-isolation . && useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["uvicorn", "paymentguard.api:app", "--host", "0.0.0.0", "--port", "8000"]
