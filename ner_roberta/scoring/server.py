import os, sys
from fastapi import FastAPI, Request
from importlib.util import spec_from_file_location, module_from_spec
from transformers.file_utils import PaddingStrategy
import json
import torch
import typing
import time
from transformers import RobertaTokenizer
import nltk
from fastapi.middleware.cors import CORSMiddleware
from transformers.utils import logging


logger = logging.get_logger(__name__)


def load_json_configs(folder: str) -> typing.Tuple[dict, dict, dict, dict]:
    """
    Loads config, POS tags dictionary, NER tags dictionary, NER tags description json files
    from the specified folder.
    :param folder: path to the folder with needed files
    :return:
    """
    with open(os.path.join(folder, "config.json"), 'r') as file:
        config = json.load(file)
    with open(os.path.join(folder, "pos_tags_dict.json"), 'r') as file:
        pos_tags_dict = json.load(file)
    with open(os.path.join(folder, "ner_tags_dict.json"), 'r') as file:
        ner_tags_dict = json.load(file)
    with open(os.path.join(folder, "ner_description.json"), 'r') as file:
        ner_description_dict = json.load(file)
    return config, pos_tags_dict, ner_tags_dict, ner_description_dict


def load_model(folder: str, config: dict) -> torch.nn.Module:
    """
    Loads the model class and weights from the specified folder.

    :param folder:
    :param config:
    :return:
    """
    spec = spec_from_file_location("ner_score_module", os.path.join(folder, "score_model.py"))
    model_module = module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model_module.__init__("ner_score_module")
    model_object: torch.nn.Module = model_module.RobertaNerScore(**config)
    state_dict = torch.load(os.path.join(folder, 'model.pt'))
    model_object.load_state_dict(state_dict)
    return model_object


def prepare_text(text, tokenizer, pos_tags_dict, json_config):
    DEFAULT_SENTENCE_LEN = json_config['default_sentence_len']
    text = " ".join(nltk.tokenize.word_tokenize(text))
    tokens = nltk.tokenize.word_tokenize(text)
    pos_tags = [pos_tags_dict.get(item[1], pos_tags_dict[json_config["config.POS.UNK_POS_TAG"]]) for item in
                nltk.pos_tag(tokens)]
    upcase_fracture = []
    numbers_fracture = []
    for token in tokens:
        token_upcase_count = 0
        numbers_count = 0
        for char in token:
            if char.isupper():
                token_upcase_count += 1
            if char.isnumeric():
                numbers_count += 1
        upcase_fracture.append(token_upcase_count / len(token))
        numbers_fracture.append(numbers_count / len(token))
    upcase_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(upcase_fracture))
    numbers_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(numbers_fracture))
    pos_tags += [pos_tags_dict[json_config["config.POS.PAD_POS_TAG"]]] * (DEFAULT_SENTENCE_LEN - len(pos_tags))
    tokenizer_output = tokenizer(text, padding=PaddingStrategy.MAX_LENGTH, truncation=True,
                                      max_length=DEFAULT_SENTENCE_LEN, return_token_type_ids=True,
                                      return_attention_mask=True, return_special_tokens_mask=True)
    input_ids = tokenizer_output['input_ids']
    token_type_ids = tokenizer_output['token_type_ids']
    attention_mask = tokenizer_output['attention_mask']

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    upcase_fracture = torch.tensor(upcase_fracture, dtype=torch.float).unsqueeze(0)
    numbers_fracture = torch.tensor(numbers_fracture, dtype=torch.float).unsqueeze(0)
    pos_tags = torch.tensor(pos_tags, dtype=torch.long).unsqueeze(0)
    tokens_count = len(tokens)
    return (input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags), tokens_count, tokens

def make_prediction(model, input_data, tokens_count, tokens, ner_tags_list, tags_to_remove):
    output = model(*input_data)
    for tag_index in tags_to_remove:
        output[:, :, tag_index] = -1000
    ner_indices = output.max(dim=2).indices.flatten().tolist()
    ner_indices = ner_indices[1: tokens_count + 1]
    ner_tags = [ner_tags_list[index] for index in ner_indices]
    results = [{"Token" : token, "Tag" : ner_tag} for token, ner_tag in zip(tokens, ner_tags)]
    return results


def startup_server():
    logger.log(logging.INFO, "Starting up the server")
    package_dir = os.environ['PACKAGE_DIR']
    config, pos_tags_dict, ner_tags_dict, ner_description_dict = load_json_configs(package_dir)
    logger.log(logging.INFO, "Json dictionaries loaded")
    model = load_model(package_dir, config)
    logger.log(logging.INFO, "Model loaded")
    # os.environ['NLTK'] = os.path.join(package_dir, "NLTK")
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(package_dir, "tokenizer"))
    logger.log(logging.INFO, "Loaded the RoBERTa tokenizer")
    ner_tags_list = sorted(ner_tags_dict, key=lambda x: ner_tags_dict[x])
    tags_to_remove = config["config.SCORE.TAGS_TO_REMOVE"]
    tags_to_remove = [ner_tags_dict[tag] for tag in tags_to_remove]
    logger.log(logging.INFO, "Finished server start up")

    return model, tokenizer, pos_tags_dict, ner_tags_dict, ner_tags_list, config, ner_description_dict, tags_to_remove


model, tokenizer, pos_tags_dict, ner_tags_dict, ner_tags_list, json_config, ner_description_dict, tags_to_remove = startup_server()

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/prediction")
async def prediction(request: Request):
    start = time.time()
    body = await request.json()
    input_data, tokens_count, tokens = prepare_text(body['text'], tokenizer,
                                                    pos_tags_dict, json_config)
    result = make_prediction(model, input_data, tokens_count, tokens, ner_tags_list, tags_to_remove)
    logger.log(logging.INFO, f"Prediction was made in {time.time() - start} seconds")
    return {"prediction": result}

@app.get("/ner_description")
async def ner_description():
    logger.log(logging.INFO, "Returning NER tags description")
    return {"description": ner_description_dict}



