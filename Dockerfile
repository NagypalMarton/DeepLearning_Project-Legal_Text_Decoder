FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

# Munkakönyvtár beállítása
WORKDIR /app

# A data és output könyvtárak létrehozása és használatuk beállítása
RUN mkdir -p /app/data /app/output
ENV DATA_DIR=/app/data \
    OUTPUT_DIR=/app/output
VOLUME ["/app/data", "/app/output"]

# Python függőségek másolása és telepítése (jobb cache rétegezés)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Alkalmazás kódjának másolása
COPY ./ .

# A futtató script végrehajthatóvá tétele
RUN chmod +x src/run.sh

# Alapértelmezett parancs a konténer indításakor (futtatáskor, nem build-kor)
CMD ["bash", "src/run.sh"]