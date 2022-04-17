from ner_roberta.training.config import MainConfig
import os
import shutil
from ner_roberta.training.model import RobertaNER, build_model_from_train_checkpoint
from transformers import RobertaTokenizer
import torch
import json
import nltk
from pathlib import Path



def get_project_root() -> Path:
    return Path(__file__).parent.parent.parent

def load_tags_dictionaries(config):
    with open(config.POS.POS_TAGS_DICT_FILEPATH, 'r') as file:
        pos_tags_dict = json.load(file)

    with open(config.NER.NER_TAGS_DICT_FILEPATH, 'r') as file:
        ner_tags_dict = json.load(file)
    return pos_tags_dict, ner_tags_dict

def clear_folder_contents(folder):
    stop_counter = 10
    while len(os.listdir(folder)) > 0:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        stop_counter -= 1
        if stop_counter < 1:
            raise Exception("Could not delete all files")


def build_output_package_for_fast_api(trained_model: RobertaNER, config: MainConfig):
    project_root = get_project_root()
    package_folder = os.path.join(project_root, config.SCORE.PACKAGE_FOLDER)
    print("Clearing output folder contents")
    clear_folder_contents(package_folder)
    print("Loading json files")
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config)
    json_config = {
        "pos_tags_count": len(pos_tags_dict),
        "pos_embeddings_size": config.MODEL.POS_EMBEDDINGS_SIZE,
        "ner_tags_count": len(ner_tags_dict),
        "default_sentence_len": config.MODEL.DEFAULT_SENTENCE_LEN,
        "config.POS.UNK_POS_TAG": config.POS.UNK_POS_TAG,
        "config.POS.PAD_POS_TAG": config.POS.PAD_POS_TAG,
    }

    with open(os.path.join(package_folder, "config.json"), 'w') as file:
        json.dump( json_config, file)

    shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/score_model.py"),  os.path.join(package_folder, 'score_model.py'),)
    shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/server.py"), os.path.join(package_folder, 'server.py'), )

    nltk_path = os.path.join(package_folder, "NLTK")
    if not os.path.isdir( os.path.join(package_folder, "NLTK") ):
        os.mkdir(nltk_path)
    nltk.download('punkt', download_dir=nltk_path)
    nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)
    print("Copying json files")
    shutil.copyfile(config.POS.POS_TAGS_DICT_FILEPATH, os.path.join(package_folder, os.path.basename(config.POS.POS_TAGS_DICT_FILEPATH)) )
    shutil.copyfile(config.NER.NER_TAGS_DICT_FILEPATH, os.path.join(package_folder, os.path.basename(config.NER.NER_TAGS_DICT_FILEPATH)))
    shutil.copyfile(config.SCORE.NER_DESCRIPTION_DICTIONARY_PATH, os.path.join(package_folder, os.path.basename(config.SCORE.NER_DESCRIPTION_DICTIONARY_PATH)))

    print("Saving tokenizer")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(os.path.join(package_folder, "tokenizer"))
    print("Saving model")
    with open(os.path.join(package_folder, "config.json"), 'w') as file:
        json.dump( json_config, file)
    torch.save(trained_model.state_dict(), os.path.join(package_folder, 'model.pt'))
    requirements = [
        "fastapi",
        "uvicorn",
        "transformers",
        "torch",
        "pydantic",
        "nltk",
    ]
    with open(os.path.join(package_folder, "requirements.txt"), 'w') as file:
        file.writelines( ["\n" + line for line in requirements] )


if __name__ == '__main__':
    from ner_roberta.training.config import get_config
    config = get_config()
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config)
    model_path = os.path.join(config.TRAIN.START_TRAIN_CHECKPOINT, "pytorch_model.bin")
    model = build_model_from_train_checkpoint(ner_tags_dict, len(pos_tags_dict), config, model_path)
    build_output_package_for_fast_api(model, config)