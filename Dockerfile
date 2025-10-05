FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# A data és output könyvtárak létrehozása és használatuk beállítása
RUN mkdir -p /app/data /app/output
ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output
VOLUME ["/app/data", "/app/output"]

COPY ./ ./
RUN pip install --no-cache-dir -r ./requirements.txt

RUN ./src/run.sh