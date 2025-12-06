FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Munkakönyvtár beállítása
WORKDIR /app

# A data és output könyvtárak létrehozása és használatuk beállítása
RUN mkdir -p /app/data /app/output
ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output \
    START_API_SERVICE=1 \
    START_FRONTEND_SERVICE=1 \
    API_URL=http://localhost:8000 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

VOLUME ["/app/data", "/app/output"]

# Python függőségek másolása és telepítése (jobb cache rétegezés)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Alkalmazás kódjának másolása
COPY ./ .

# A futtató script végrehajthatóvá tétele
RUN chmod +x src/run.sh

# Healthcheck (opcionális, de hasznos)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import sys; sys.exit(0)" || exit 1

# Alapértelmezett parancs a konténer indításakor (futtatáskor, nem build-kor)
CMD ["bash", "src/run.sh"]