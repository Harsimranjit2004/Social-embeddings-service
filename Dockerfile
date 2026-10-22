# Install Python dependencies in a builder and copy only those packages into
# the minimal runtime. Pin both multi-platform base indexes for reproducible builds.
FROM python:3.11-slim-bookworm@sha256:f06d01d4ba3f8ad5af45a3ec87df19c1d317d8e8961edc0fabda61811af6f7cf AS build
WORKDIR /build
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --no-compile --target=/opt/python -r requirements.txt

FROM gcr.io/distroless/python3-debian12:nonroot@sha256:7d1042ce588ab97019fe95c24ffca7bc5a82ccdac572511d5e09bda4435c89c5
ENV PYTHONPATH=/opt/python \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_DIR=/app/model \
    ORT_INTRA_OP_THREADS=2
WORKDIR /app
COPY --from=build /opt/python /opt/python
COPY --chown=10001:10001 app ./app
COPY --chown=10001:10001 model ./model
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
  CMD ["python3", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)"]
CMD ["-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
