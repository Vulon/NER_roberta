FROM pytorch/torchserve:latest
USER root
RUN pip install --upgrade pip
COPY model_package/requirements.txt /home/model-server/
RUN pip install -r /home/model-server/requirements.txt

COPY model_package/model.pt /home/model-server/
COPY ner_roberta/scoring/score.py /home/model-server/
COPY ner_roberta/scoring/handler.py /home/model-server/
COPY model_package/config.json /home/model-server/
COPY model_package/ner_tags_dict.json /home/model-server/
COPY model_package/pos_tags_dict.json /home/model-server/


RUN printf "\nservice_envelope=json" >> /home/model-server/config.properties
USER model-server

EXPOSE 8080
EXPOSE 8081
EXPOSE 8082

RUN torch-model-archiver -f \
  --model-name=ner_roberta \
  --version=1.0 \
  --serialized-file=/home/model-server/model.pt \
  --model-file=/home/model-server/score.py \
  --handler=/home/model-server/handler.py \
  --extra-files "/home/model-server/config.json,/home/model-server/ner_tags_dict.json,/home/model-server/pos_tags_dict.json" \
  --export-path=/home/model-server/model-store


CMD ["torchserve", \
     "--start", \
     "--no-config-snapshots", \
     "--ts-config=/home/model-server/config.properties", \
     "--models", \
     "ner_roberta=ner_roberta.mar", \
     "--model-store", \
     "/home/model-server/model-store"]