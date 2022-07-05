FROM python:3.10.5-slim-bullseye
USER root

RUN mkdir /home/model-server
WORKDIR /home/model-server/


COPY ["model_package/Pipfile", "model_package/Pipfile.lock", "/home/model-server/"]
RUN pip3 install --no-cache-dir pipenv
RUN pipenv install --system --deploy

COPY model_package/NLTK /root/nltk_data
COPY model_package/tokenizer /home/model-server/tokenizer
COPY model_package/model.pt /home/model-server/

COPY [ "model_package/config.json", "model_package/ner_tags_dict.json", "model_package/pos_tags_dict.json", "model_package/ner_description.json", "model_package/test_examples.json", "model_package/google.json", "model_package/score_model.py", "model_package/server.py", "/home/model-server/"]

ENV PACKAGE_DIR=/home/model-server/

EXPOSE 7000

CMD ["uvicorn", "server:app", "--port", "7000", "--host", "0.0.0.0"]