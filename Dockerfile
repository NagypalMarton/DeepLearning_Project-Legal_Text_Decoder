FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

COPY ./ ./
RUN pip install --no-cache-dir -r ./requirements.txt

RUN ./src/run.sh