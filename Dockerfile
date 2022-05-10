FROM pytorch/torchserve:latest
USER root
#RUN mkdir /home/model-server/
RUN rm -r /home/model-server/*

COPY model_package/model.pt /home/model-server/
COPY model_package/requirements.txt /home/model-server/
RUN pip3 install --no-cache-dir -r /home/model-server/requirements.txt

COPY model_package/NLTK /root/nltk_data
COPY model_package/tokenizer /home/model-server/tokenizer
COPY model_package/config.json /home/model-server/
COPY model_package/ner_tags_dict.json /home/model-server/
COPY model_package/pos_tags_dict.json /home/model-server/
COPY model_package/ner_description.json /home/model-server/
COPY model_package/test_examples.json /home/model-server/
COPY model_package/google.json /home/model-server/
COPY model_package/score_model.py /home/model-server/
COPY model_package/server.py /home/model-server/

ENV PACKAGE_DIR=/home/model-server/

EXPOSE 7000

WORKDIR /home/model-server/

CMD ["uvicorn", "server:app", "--port", "7000", "--host", "0.0.0.0"]