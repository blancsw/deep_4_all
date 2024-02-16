FROM pytorch/torchserve:0.9.0-cpu
# Setup dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.properties /home/model-server/config.properties
COPY handler.py .
RUN torch-model-archiver --model-name lang_detect --version 1.0.0 --serialized-file handler.py --handler handler.py \
    && mv lang_detect.mar /home/model-server/model-store