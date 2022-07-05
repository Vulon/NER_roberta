import json
import shutil
import torch
import os, sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from ner_roberta.training.model import RobertaNER, build_model_from_train_checkpoint
from ner_roberta.miscellaneous.utils import load_tags_dictionaries, extract_test_examples
from transformers import RobertaTokenizer

def build_output_package_for_fast_api(config: dict):
    project_root = os.environ["DVC_ROOT"]

    pos_tags_dict_filepath = os.path.join(project_root, config["tags"]["POS"]["POS_TAGS_DICT_FILEPATH"])
    ner_tags_dict_filepath = os.path.join(project_root, config["tags"]["NER"]["NER_TAGS_DICT_FILEPATH"])
    ner_description_filepath = os.path.join(project_root, config["SCORE"]["NER_DESCRIPTION_DICTIONARY_PATH"])

    package_folder = os.path.join(project_root, config["SCORE"]["PACKAGE_FOLDER"])
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(pos_tags_dict_filepath, ner_tags_dict_filepath)

    model_path = os.path.join("output/model", "pytorch_model.bin")
    model = build_model_from_train_checkpoint(len(ner_tags_dict), len(pos_tags_dict), config["MODEL"], model_path)

    score_config = {
        "pos_tags_count": len(pos_tags_dict),
        "ner_tags_count": len(ner_tags_dict),
        "POS_EMBEDDINGS_SIZE": config["MODEL"]["POS_EMBEDDINGS_SIZE"],
        "DEFAULT_SENTENCE_LEN": config["MODEL"]["DEFAULT_SENTENCE_LEN"],
        "UNK_POS_TAG": config["tags"]["POS"]["UNK_POS_TAG"],
        "PAD_POS_TAG": config["tags"]["POS"]["PAD_POS_TAG"],
        "TAGS_TO_REMOVE": config["SCORE"]["TAGS_TO_REMOVE"],
        "MAX_BATCH_SIZE": config["SCORE"]["MAX_BATCH_SIZE"],
        "MAX_OPTIMAL_SENTENCE_SIZE": config["SCORE"]["MAX_OPTIMAL_SENTENCE_SIZE"],
    }

    test_examples = extract_test_examples(config["DATA"]["TEST_DATASET_PATH"], config["MODEL"]["TOKENIZER_NAME"], config["SCORE"]["TEST_EXAMPLE_INDICES"])

    with open( os.path.join(package_folder, config["SCORE"]["TEST_EXAMPLES_FILE_NAME"]) , "w") as file:
        json.dump(test_examples, file)
    with open(os.path.join(package_folder, "config.json"), 'w') as file:
        json.dump( score_config, file)

    shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/score_model.py"),  os.path.join(package_folder, 'score_model.py'),)
    shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/server.py"), os.path.join(package_folder, 'server.py'), )

    shutil.copyfile(pos_tags_dict_filepath, os.path.join(package_folder, os.path.basename(pos_tags_dict_filepath)) )
    shutil.copyfile(ner_tags_dict_filepath, os.path.join(package_folder, os.path.basename(ner_tags_dict_filepath)))
    shutil.copyfile(ner_description_filepath, os.path.join(package_folder, os.path.basename(ner_description_filepath)))
    shutil.copyfile(os.path.join(project_root, "keys/google.json"), os.path.join(package_folder, "google.json"))

    tokenizer = RobertaTokenizer.from_pretrained(config["MODEL"]["TOKENIZER_NAME"])
    tokenizer.save_pretrained(os.path.join(package_folder, "tokenizer"))
    torch.save(model.state_dict(), os.path.join(package_folder, 'model.pt'))
    shutil.copyfile(os.path.join(project_root, "Pipfile"), os.path.join(package_folder, "Pipfile"))
    shutil.copyfile(os.path.join(project_root, "Pipfile.lock"), os.path.join(package_folder, "Pipfile.lock"))



if __name__ == "__main__":
    params = dvc.api.params_show()
    build_output_package_for_fast_api(params)
