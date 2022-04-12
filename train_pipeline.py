import warnings
warnings.filterwarnings('ignore')
from transformers import RobertaForMaskedLM, RobertaConfig
import pickle
from config import get_config
from model import RobertaNER
from metrics import cross_entropy_with_attention, create_compute_metrics_function
import json
import torch
import numpy as np
import random
from transformers import Trainer, TrainingArguments
import torch.nn as nn
from dataset import load_tags_dictionaries
from utils import build_output_package

def set_random_seed():
    np.random.seed(config.RANDOM_SEED)
    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)



config = get_config()


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
    pos_tags_dict, ner_tags_dict = load_tags_dictionaries(config)
    model = build_model(pos_tags_dict, ner_tags_dict)
    train_dataset, val_dataset, test_dataset  = load_datasets()
    compute_metrics = create_compute_metrics_function(ner_tags_dict, config)

    args = TrainingArguments(
        output_dir='output',
        evaluation_strategy='steps',
        eval_steps=config.TRAIN.EVAL_STEPS,
        per_device_train_batch_size=config.TRAIN.TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=config.TRAIN.VAL_BATCH_SIZE,
        num_train_epochs=config.TRAIN.TRAIN_EPOCHS,
        seed=config.RANDOM_SEED,
        save_steps=config.TRAIN.SAVE_STEPS,
        #     bf16=True,
        fp16=config.TRAIN.USE_FLOAT_PRECISION_16,
        gradient_accumulation_steps=config.TRAIN.GRADIENT_ACCUMULATION_STEPS,
        eval_accumulation_steps = config.TRAIN.EVAL_ACCUMULATION_STEPS,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    build_output_package(model, config)
