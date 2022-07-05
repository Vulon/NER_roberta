import json
import warnings
import os, sys
import dvc.api
sys.path.append(os.environ["DVC_ROOT"])
import transformers
warnings.filterwarnings('ignore')
from transformers import RobertaForMaskedLM
import pickle
from ner_roberta.training.model import RobertaNER
from ner_roberta.training.metrics import cross_entropy_with_attention, create_compute_metrics_function
import torch
import numpy as np
import random
from transformers import Trainer, TrainingArguments
from ner_roberta.miscellaneous.utils import load_tags_dictionaries, build_model_from_train_checkpoint
from transformers.utils import logging
import dvc.api

logger = logging.get_logger(__name__)

def set_random_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def load_datasets(train_filepath, val_filepath, test_filepath ):
    with open(train_filepath, 'rb') as file:
        train_dataset = pickle.load(file)

    with open(val_filepath, 'rb') as file:
        val_dataset = pickle.load(file)

    with open(test_filepath, 'rb') as file:
        test_dataset = pickle.load(file)
    return train_dataset, val_dataset, test_dataset


if __name__ == '__main__':
    params = dvc.api.params_show()
    set_random_seed(params["RANDOM_SEED"])
    logger.log(logging.INFO, "Starting the training")

    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(params['tags']['POS']["POS_TAGS_DICT_FILEPATH"], params['tags']['NER']["NER_TAGS_DICT_FILEPATH"])
    logger.log(logging.INFO, "Loaded NER and POS dicts")

    model = build_model_from_train_checkpoint(len(ner_tags_dict), len(pos_tags_dict), params['MODEL'])
    logger.log(logging.INFO, "Loaded model weights")
    train_dataset, val_dataset, test_dataset = load_datasets(params["DATA"]["TRAIN_DATASET_PATH"], params["DATA"]["VAL_DATASET_PATH"], params["DATA"]["TEST_DATASET_PATH"])
    logger.log(logging.INFO, "Loaded the datasets")
    compute_metrics = create_compute_metrics_function(ner_tags_dict, params["tags"]["NER"]["CLS_TAG"])


    args = TrainingArguments(
        output_dir='output',
        evaluation_strategy='steps',
        eval_steps=params["TRAIN"]["EVAL_STEPS"],
        per_device_train_batch_size=params["TRAIN"]["TRAIN_BATCH_SIZE"],
        per_device_eval_batch_size=params["TRAIN"]["VAL_BATCH_SIZE"],
        num_train_epochs=params["TRAIN"]["TRAIN_EPOCHS"],
        seed=params["RANDOM_SEED"],
        save_steps=params["TRAIN"]["SAVE_STEPS"],
        fp16=params["TRAIN"]["USE_FLOAT_PRECISION_16"],
        gradient_accumulation_steps=params["TRAIN"]["GRADIENT_ACCUMULATION_STEPS"],
        eval_accumulation_steps=params["TRAIN"]["EVAL_ACCUMULATION_STEPS"],
        learning_rate = params["TRAIN"]["LEARNING_RATE"],
        weight_decay = params["TRAIN"]["WEIGHTS_DECAY"]
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,

    )
    checkpoint = params["TRAIN"]["START_TRAIN_CHECKPOINT"] if params["TRAIN"]["START_TRAIN_CHECKPOINT"] else None
    trainer.train(checkpoint)
    logger.log(logging.INFO, "Training finished")
    val_metrics = trainer.evaluate(val_dataset)
    test_metrics = trainer.evaluate(test_dataset)
    with open("output/val_metrics.json", 'w') as file:
        json.dump(val_metrics, file)
    with open("output/test_metrics.json", 'w') as file:
        json.dump(test_metrics, file)
    trainer.save_model("output/model")