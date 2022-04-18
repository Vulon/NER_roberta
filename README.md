<h1>Named Entity Recognition webservice</h1>
<h3>Based on pretrained RoBERTa</h3>
<br>
The <b>Website</b> with API demo can be accessed via <a href="https://pos-taggers.de.r.appspot.com/">this link</a>
The Frontend is written in React, deployed in Google Cloud.
<br>
<br>
The <b>Backend</b> is deployed in Google Cloud Platform. 
API can be accessed via <a href="https://ner-roberta-uca55u3kfq-ew.a.run.app/prediction">this link</a>
<br>
Server listens to POST requests with "text" field in the request body
<br>
The model is trained on 
<a href="https://www.kaggle.com/datasets/naseralqaydeh/named-entity-recognition-ner-corpus">this corpus</a>

The model is based on "roberta-base" from the <a href="https://huggingface.co/roberta-base">Hugging Face 🤗</a>
The Language modeling head was replaced with simple linear layer for predicting NER tags.
Additionally, the NER head takes POS tags embeddings and percentage of uppercase characters as the input. 

