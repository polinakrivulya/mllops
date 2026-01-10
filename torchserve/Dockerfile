FROM pytorch/torchserve:latest

USER root

RUN pip install --no-cache-dir "transformers>=4.44" "tokenizers>=0.19" "numpy>=1.24"

WORKDIR /home/model-server

COPY torchserve/model-store /home/model-server/model-store

CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--model-store", "model-store", \
     "--models", "mymodel=mymodel.mar", \
     "--foreground"]
