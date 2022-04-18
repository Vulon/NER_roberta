import json
import warnings
import os
import transformers
warnings.filterwarnings('ignore')
from transformers import RobertaForMaskedLM
import pickle
from ner_roberta.training.config import get_config
from ner_roberta.training.model import RobertaNER
from ner_roberta.training.metrics import cross_entropy_with_attention, create_compute_metrics_function
import torch
import numpy as np
import random
from transformers import Trainer, TrainingArguments
from ner_roberta.miscellaneous.utils import load_tags_dictionaries, build_output_package_for_fast_api
from transformers.utils import logging


logger = logging.get_logger(__name__)
config = get_config()


def set_random_seed():
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)


def load_datasets():
    with open(config.TRAIN.TRAIN_DATASET_PATH, 'rb') as file:
        train_dataset = pickle.load(file)

    with open(config.TRAIN.VAL_DATASET_PATH, 'rb') as file:
        val_dataset = pickle.load(file)

    with open(config.TRAIN.TEST_DATASET_PATH, 'rb') as file:
        test_dataset = pickle.load(file)
    return train_dataset, val_dataset, test_dataset


def build_model(pos_tags_dict, ner_tags_dict):
    roberta_base = RobertaForMaskedLM.from_pretrained(config.MODEL.PRETRAINED_MODEL_NAME)
    ner_roberta = RobertaNER(roberta_base, ner_tags_dict, len(pos_tags_dict), config, cross_entropy_with_attention)

    return ner_roberta


if __name__ == '__main__':
    set_random_seed()
    logger.log(logging.INFO, "Starting the training")
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config)
    logger.log(logging.INFO, "Loaded NER and POS dicts")
    model = build_model(pos_tags_dict, ner_tags_dict)
    logger.log(logging.INFO, "Loaded model weights")
    train_dataset, val_dataset, test_dataset = load_datasets()
    logger.log(logging.INFO, "Loaded the datasets")
    compute_metrics = create_compute_metrics_function(ner_tags_dict, config)

    args = TrainingArguments(
        output_dir='../../output',
        evaluation_strategy='steps',
        eval_steps=config.TRAIN.EVAL_STEPS,
        per_device_train_batch_size=config.TRAIN.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN.VAL_BATCH_SIZE,
        num_train_epochs=config.TRAIN.TRAIN_EPOCHS,
        seed=config.RANDOM_SEED,
        save_steps=config.TRAIN.SAVE_STEPS,
        fp16=config.TRAIN.USE_FLOAT_PRECISION_16,
        gradient_accumulation_steps=config.TRAIN.GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps = config.TRAIN.EVAL_ACCUMULATION_STEPS,
        learning_rate = config.TRAIN.LEARNING_RATE,
        weight_decay = config.TRAIN.WEIGHTS_DECAY
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(config.TRAIN.START_TRAIN_CHECKPOINT) #
    logger.log(logging.INFO, "Training finished")
    build_output_package_for_fast_api(model, config)
    logger.log(logging.INFO, "Created output package")
