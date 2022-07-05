import os, sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
from ner_roberta.training.model import RobertaNER, build_model_from_train_checkpoint
from ner_roberta.miscellaneous.utils import load_tags_dictionaries, extract_test_examples

def build_output_package_for_fast_api(trained_model: RobertaNER, config: dict):

    project_root = os.environ["DVC_ROOT"]
    package_folder = os.path.join(project_root, config["SCORE"]["PACKAGE_FOLDER"])
    config["tags"]["POS"]["POS_TAGS_DICT_FILEPATH"]
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config["tags"]["POS"]["POS_TAGS_DICT_FILEPATH"], config["tags"]["NER"]["NER_TAGS_DICT_FILEPATH"])

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

    test_examples = extract_test_examples(config["DATA"]["TEST_DATASET_PATH"])


if __name__ == "__main__":
    params = dvc.api.params_show()

# def build_output_package_for_fast_api(trained_model: RobertaNER, config: MainConfig):
#     project_root = get_project_root()
#     package_folder = os.path.join(project_root, config.SCORE.PACKAGE_FOLDER)
#     print("Clearing output folder contents")
#     clear_folder_contents(package_folder)
#     print("Loading json files")
#     pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config)
#     json_config = {
#         "pos_tags_count": len(pos_tags_dict),
#         "pos_embeddings_size": config.MODEL.POS_EMBEDDINGS_SIZE,
#         "ner_tags_count": len(ner_tags_dict),
#         "default_sentence_len": config.MODEL.DEFAULT_SENTENCE_LEN,
#         "config.POS.UNK_POS_TAG": config.POS.UNK_POS_TAG,
#         "config.POS.PAD_POS_TAG": config.POS.PAD_POS_TAG,
#         "config.SCORE.TAGS_TO_REMOVE": config.SCORE.TAGS_TO_REMOVE,
#         "config.SCORE.MAX_BATCH_SIZE": config.SCORE.MAX_BATCH_SIZE,
#         "config.SCORE.MAX_OPTIMAL_SENTENCE_SIZE": config.SCORE.MAX_OPTIMAL_SENTENCE_SIZE,
#     }
#
#     test_examples = extract_test_examples(config, config.SCORE.TEST_EXAMPLE_INDICES)
#
#     with open( os.path.join(package_folder, config.SCORE.TEST_EXAMPLES_FILE_NAME) , "w") as file:
#         json.dump(test_examples, file)
#
#
#     with open(os.path.join(package_folder, "config.json"), 'w') as file:
#         json.dump( json_config, file)
#
#     shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/score_model.py"),  os.path.join(package_folder, 'score_model.py'),)
#     shutil.copyfile(os.path.join(project_root, "ner_roberta/scoring/server.py"), os.path.join(package_folder, 'server.py'), )
#
#     nltk_path = os.path.join(package_folder, "NLTK")
#     if not os.path.isdir( os.path.join(package_folder, "NLTK") ):
#         os.mkdir(nltk_path)
#     nltk.download('punkt', download_dir=nltk_path)
#     nltk.download('averaged_perceptron_tagger', download_dir=nltk_path)
#     print("Copying json files")
#     shutil.copyfile(config.POS.POS_TAGS_DICT_FILEPATH, os.path.join(package_folder, os.path.basename(config.POS.POS_TAGS_DICT_FILEPATH)) )
#     shutil.copyfile(config.NER.NER_TAGS_DICT_FILEPATH, os.path.join(package_folder, os.path.basename(config.NER.NER_TAGS_DICT_FILEPATH)))
#     shutil.copyfile(config.SCORE.NER_DESCRIPTION_DICTIONARY_PATH, os.path.join(package_folder, os.path.basename(config.SCORE.NER_DESCRIPTION_DICTIONARY_PATH)))
#
#     shutil.copyfile(os.path.join(project_root, "keys/google.json"), os.path.join(package_folder, "google.json"))
#
#     print("Saving tokenizer")
#     tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#     tokenizer.save_pretrained(os.path.join(package_folder, "tokenizer"))
#     print("Saving model")
#     with open(os.path.join(package_folder, "config.json"), 'w') as file:
#         json.dump( json_config, file)
#     torch.save(trained_model.state_dict(), os.path.join(package_folder, 'model.pt'))
#     requirements = [
#         "fastapi",
#         "uvicorn",
#         "transformers",
#         "torch",
#         "pydantic",
#         "nltk",
#         "google-cloud-logging"
#     ]
#     with open(os.path.join(package_folder, "requirements.txt"), 'w') as file:
#         file.writelines( ["\n" + line for line in requirements] )