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


# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.environ['PACKAGE_DIR'], "google.json")
# import google.cloud.logging
# gc_logger = google.cloud.logging.Client()
# gc_logger.setup_logging()
import logging


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


def load_test_examples(folder: str) -> list:
    with open(os.path.join(folder, "test_examples.json"), 'r') as file:
        test_examples = json.load(file)
    return test_examples


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
    model_object: torch.nn.Module = model_module.RobertaNerScore(
        config["pos_tags_count"],
        config["POS_EMBEDDINGS_SIZE"],
        config["ner_tags_count"],
        config["DEFAULT_SENTENCE_LEN"],
    )
    state_dict = torch.load(os.path.join(folder, 'model.pt'))
    model_object.load_state_dict(state_dict)
    return model_object


def get_tokenizer_tensors(line: str, DEFAULT_SENTENCE_LEN: int):
    tokenizer_output = tokenizer(line, padding=PaddingStrategy.MAX_LENGTH, truncation=True,
                                 max_length=DEFAULT_SENTENCE_LEN, return_token_type_ids=True,
                                 return_attention_mask=True, return_special_tokens_mask=True)
    input_ids = tokenizer_output['input_ids']
    token_type_ids = tokenizer_output['token_type_ids']
    attention_mask = tokenizer_output['attention_mask']

    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long).unsqueeze(0)
    return input_ids, token_type_ids, attention_mask


def extract_character_level_tensors(tokens: typing.List[str], DEFAULT_SENTENCE_LEN: int):
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
        token_upcase_count /= len(token)
        numbers_count /= len(token)
        upcase_fracture.append(token_upcase_count)
        numbers_fracture.append(numbers_count)
    upcase_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(upcase_fracture))
    numbers_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(numbers_fracture))
    upcase_fracture = torch.tensor(upcase_fracture, dtype=torch.float).unsqueeze(0)
    numbers_fracture = torch.tensor(numbers_fracture, dtype=torch.float).unsqueeze(0)
    return upcase_fracture, numbers_fracture


def extract_pos_tags_tensors(tokens : typing.List[str], DEFAULT_SENTENCE_LEN: int, json_config: dict):
    pos_tags = [pos_tags_dict.get(item[1], pos_tags_dict[json_config["UNK_POS_TAG"]]) for item in
                nltk.pos_tag(tokens)]
    pos_tags += [pos_tags_dict[json_config["PAD_POS_TAG"]]] * (DEFAULT_SENTENCE_LEN - len(pos_tags))
    pos_tags = torch.tensor(pos_tags, dtype=torch.long).unsqueeze(0)
    return pos_tags


def prepare_tensors_for_scoring(line : str, DEFAULT_SENTENCE_LEN: int, json_config: dict):
    tokens = nltk.tokenize.word_tokenize(line)
    tokens_count = len(tokens)

    input_ids, token_type_ids, attention_mask = get_tokenizer_tensors(line, DEFAULT_SENTENCE_LEN)
    upcase_fracture, numbers_fracture = extract_character_level_tensors(tokens, DEFAULT_SENTENCE_LEN)
    pos_tags = extract_pos_tags_tensors(tokens, DEFAULT_SENTENCE_LEN, json_config)
    tensors_list = (
        input_ids, token_type_ids, attention_mask,
        upcase_fracture, numbers_fracture,
        pos_tags
    )

    return tensors_list, tokens, tokens_count


def arrange_text(text: str, json_config: dict) -> typing.List[str]:
    max_optimal_len = json_config['MAX_OPTIMAL_SENTENCE_SIZE']
    temp_lines = nltk.sent_tokenize(text)
    lines_buffer = ""
    text_lines = []
    for line in temp_lines:
        if len(line) > max_optimal_len:
            line = line[ : max_optimal_len]
            text_lines.append(lines_buffer)
            text_lines.append(line)
            lines_buffer = ""
            continue

        candidate = lines_buffer + " " + line
        if len(candidate) > max_optimal_len:
            text_lines.append(lines_buffer)
            lines_buffer = line
        else:
            lines_buffer = candidate
    text_lines.append(lines_buffer)

    text_lines = [ " ".join(nltk.tokenize.word_tokenize(line)) for line in text_lines ]
    return text_lines


def process_model_output(output : torch.Tensor, tokens_count_list: typing.List[int], tokens_list: typing.List[typing.List[str]] ,  tags_to_remove: typing.List[int], ner_tags_list: list):
    for tag_index in tags_to_remove:
        output[:, :, tag_index] = -1000
    ner_indices = output.max(dim=2).indices
    results = []
    for i in range(ner_indices.shape[0]):
        tokens_count = tokens_count_list[i]
        tokens = tokens_list[i]
        line_predictions = ner_indices[i, 1 : tokens_count + 1]
        print(tokens)
        print(line_predictions)
        line_tags = [ner_tags_list[index] for index in line_predictions]
        results.append(
            [{"Token" : token, "Tag" : ner_tag} for token, ner_tag in zip(tokens, line_tags)]
        )
    return results


def make_single_batch_prediction(lines: typing.List[str], model: torch.nn.Module, tags_to_remove: typing.List[int], json_config: dict, ner_tags_list: list):
    DEFAULT_SENTENCE_LEN = json_config['DEFAULT_SENTENCE_LEN']
    results = []
    logging.debug(f"Predicting {len(lines)} lines")
    for line in lines:
        tensors_list, tokens, tokens_count = prepare_tensors_for_scoring(line, DEFAULT_SENTENCE_LEN, json_config)
        output = model(*tensors_list)
        result = process_model_output(output, [tokens_count], [tokens], tags_to_remove, ner_tags_list)[0]
        results.append(result)
    return results


def startup_server():
    start = time.time()
    global_start = start
    logging.info("Starting up the NER server")
    package_dir = os.environ['PACKAGE_DIR']
    config, pos_tags_dict, ner_tags_dict, ner_description_dict = load_json_configs(package_dir)
    logging.info(f"NER server: loaded json files in {time.time() - start} s")
    start = time.time()

    model = load_model(package_dir, config)
    logging.info(f"NER server: loaded model in {time.time() - start} s")
    start = time.time()
    tokenizer = RobertaTokenizer.from_pretrained(os.path.join(package_dir, "tokenizer"))
    logging.info(f"NER server: loaded RoBERTa tokenzer in {time.time() - start} s")

    ner_tags_list = sorted(ner_tags_dict, key=lambda x: ner_tags_dict[x])
    print(ner_tags_list)
    print(ner_tags_dict)
    tags_to_remove = config["TAGS_TO_REMOVE"]
    tags_to_remove = [ner_tags_dict[tag] for tag in tags_to_remove]
    logging.info(f"NER server: server started in {time.time() - global_start} s")

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
    text = body['text']

    text_lines = arrange_text(text, json_config)
    results = make_single_batch_prediction(text_lines, model, tags_to_remove, json_config, ner_tags_list)
    results = [ item for nested in results for item in nested ]
    logging.info(f"NER server: prediction was made in {time.time() - start} s")

    return {"prediction": results, "time": time.time() - start}


@app.post("/batch_prediction")
async def batch_prediction(request: Request):
    start = time.time()
    body = await request.json()
    text_parts = body['batch']
    del body
    final_results = []
    batch_times = []
    for raw_single_line in text_parts:
        batch_start = time.time()
        compressed_lines = arrange_text(raw_single_line, json_config)
        results = make_single_batch_prediction(compressed_lines, model, tags_to_remove, json_config, ner_tags_list)
        results = [item for nested in results for item in nested]
        final_results.append(results)
        batch_times.append(time.time() - batch_start)

    logging.info(f"NER server: prediction was made in {time.time() - start} s. Batch times: {batch_times}")
    return {"prediction": final_results,"time": time.time() - start, "batch_times" : batch_times}



@app.get("/ner_description")
async def ner_description():
    logging.info(f"NER server: NER tags description returned")
    return {"description": ner_description_dict}


@app.get("/examples")
async def examples():
    logging.info(f"NER server: examples returned")
    test_examples = load_test_examples(os.environ['PACKAGE_DIR'])
    return test_examples


