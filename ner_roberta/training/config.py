from dataclasses import dataclass
import copy

@dataclass(frozen=True)
class NerTagsConfig:
    NER_TAGS_DICT_FILEPATH: str = "../../data/ner_tags_dict.json"

    NER_TAGS_SUBSTITUTION = {
        "I-nat": 'O',
        "B-nat": 'O',
        "I-art": 'O',
        "B-art": 'O',
        "I-eve": 'O',
        "B-eve": 'O',
        "I-gpe": 'B-gpe'
    }

    START_TAG: str = "<START>"
    STOP_TAG: str = "<STOP>"
    UNK_TAG: str = "<UNK>"
    CLS_TAG: str = "CLS"


@dataclass(frozen=True)
class PosTagsConfig:
    POS_TAGS_DICT_FILEPATH: str = "../../data/pos_tags_dict.json"
    POS_TAGS_SUBSTITUTION = {
        ',': 'punkt',
        ':': 'punkt',
        '$': 'punkt',
        ')': 'punkt',
        '(': 'punkt',
        '#': 'punkt',
    }
    UNK_POS_TAG: str = "UNK"
    PAD_POS_TAG: str = "PAD"
    MINIMUM_POS_TAG_COUNT: int = 1000


@dataclass(frozen=True)
class TrainConfig:
    TRAIN_SAMPLE_FRACTURE: float = 0.8
    VAL_VS_TEST_FRACTURE: float = 0.5
    RAW_INPUT_FILEPATH: str = "../../ner.csv"
    TRAIN_DATASET_PATH: str = "../../data/train_dataset.pkl"
    VAL_DATASET_PATH: str = "../../data/val_dataset.pkl"
    TEST_DATASET_PATH: str = "../../data/test_dataset.pkl"
    EVAL_STEPS: int = 500
    TRAIN_BATCH_SIZE: int = 4
    VAL_BATCH_SIZE: int = 4
    TRAIN_EPOCHS: int = 4
    SAVE_STEPS:int = 2000
    USE_FLOAT_PRECISION_16 = True
    GRADIENT_ACCUMULATION_STEPS = 4
    EVAL_ACCUMULATION_STEPS = 4


@dataclass(frozen=True)
class ModelConfig:
    DEFAULT_SENTENCE_LEN: int = 512
    DEVICE: str = 'cuda'
    TOKENIZER_NAME: str = "roberta-base"
    PRETRAINED_MODEL_NAME: str = "roberta-base"
    POS_EMBEDDINGS_SIZE: int = 8


@dataclass(frozen=True)
class DataConfig:
    DEVICE: str = 'cpu'


@dataclass(frozen=True)
class ScoreConfig:
    PACKAGE_FOLDER: str = "model_package"
    CONFIG_FILEPATH: str = "config.py"

@dataclass(frozen=True)
class MainConfig:
    NER = NerTagsConfig()
    POS = PosTagsConfig()
    MODEL = ModelConfig()
    TRAIN = TrainConfig()
    SCORE = ScoreConfig()
    DATA = DataConfig()
    RANDOM_SEED = 42


config = MainConfig()


def get_config() -> MainConfig:
    return copy.deepcopy(config)
