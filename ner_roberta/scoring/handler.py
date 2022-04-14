from ts.torch_handler.base_handler import BaseHandler
import os
import time
import importlib.util
from ts.utils.util import list_classes_from_module
import torch
import json
from transformers import RobertaTokenizer
import nltk
from transformers.file_utils import PaddingStrategy
import logging

logger = logging.getLogger(__name__)


class NerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.pos_tags_dict = None
        self.ner_tags_dict = None
        self.mini_config = None
        self.tokenizer = None
        self.ner_tags_list = None
        logger.debug("HANDLER INIT CALLED")

    def preprocess(self, data):
        """Normalizes the input text for PyTorch model using following basic cleanup operations :
            - remove html tags
            - lowercase all text
            - expand contractions [like I'd -> I would, don't -> do not]
            - remove accented characters
            - remove punctuations
        Converts the normalized text to tensor using the source_vocab.
        Args:
            data (str): The input data is in the form of a string
        Returns:
            (Tensor): Text Tensor is returned after perfoming the pre-processing operations
            (str): The raw input is also returned in this function
        """
        logger.debug("STARTED PREPROCESSING")
        DEFAULT_SENTENCE_LEN = self.mini_config['config.MODEL.DEFAULT_SENTENCE_LEN']
        tokens = nltk.tokenize.word_tokenize(data)
        pos_tags = [self.pos_tags_dict.get(item[1], self.pos_tags_dict[self.mini_config["config.POS.UNK_POS_TAG"]]) for item in
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
        pos_tags += [self.pos_tags_dict[self.mini_config["config.POS.PAD_POS_TAG"]]] * (DEFAULT_SENTENCE_LEN - len(pos_tags))
        tokenizer_output = self.tokenizer(data, padding=PaddingStrategy.MAX_LENGTH, truncation=True,
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
        return input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags, tokens_count, tokens


    def _load_model_class(self, model_py_path: str):
        module = importlib.import_module(model_py_path.split(".")[0])
        model_class_definitions = list_classes_from_module(module)
        model_class = model_class_definitions[0]
        return model_class

    def initialize(self, context):
        """Initialize function loads the model.pt file and initialized the model object.
           First try to load torchscript else load eager mode state_dict based model.

        Args:
            context (context): It is a JSON Object containing information
            pertaining to the model artifacts parameters.

        Raises:
            RuntimeError: Raises the Runtime error when the model.py is missing
        """
        logger.debug("INITIALIZE CALLED")
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.manifest = context.manifest
        logger.debug("MODEL DIR " +  model_dir)
        with open(os.path.join(model_dir, "pos_tags_dict.json"), 'r') as file:
            self.pos_tags_dict = json.load(file)
        with open(os.path.join(model_dir, "ner_tags_dict.json"), 'r') as file:
            self.ner_tags_dict = json.load(file)
        with open(os.path.join(model_dir, "config.json"), 'r') as file:
            self.mini_config = json.load(file)

        model_file = self.manifest["model"].get("modelFile", "")
        logger.debug("MODEL FILE " + model_file)
        model_class = self._load_model_class(model_file)
        logger.debug("MODEL CLASS LOADED")
        self.model: torch.nn.Module = model_class(
            len(self.pos_tags_dict), # pos_tags_count
            self.mini_config['config.MODEL.POS_EMBEDDINGS_SIZE'], # pos_embeddings_size
            len(self.ner_tags_dict), # ner_tags_count
            self.mini_config['config.MODEL.DEFAULT_SENTENCE_LEN'],  # default_sentence_len
        )

        model_pt_path = os.path.join(model_dir, self.manifest["model"]["serializedFile"])
        model_def_path = os.path.join(model_dir, model_file)

        state_dict = torch.load(model_pt_path, map_location=self.device)
        logger.debug("STATE DICT LOADED")
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.tokenizer = RobertaTokenizer.from_pretrained( os.path.join( model_dir, "tokenizer" ) )
        logger.debug("TOKENIZER CREATED")

        self.ner_tags_list = sorted(self.ner_tags_dict, key=lambda x: self.ner_tags_dict[x])

        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')



    def postprocess(self, inference_output, tokens_count, tokens):
        """
        Return inference result.
        :param inference_output: list of inference output
        :return: list of predict results
        """
        ner_indices = inference_output.max(dim=2).indices.flatten().tolist()
        ner_indices = ner_indices[1: tokens_count + 1]
        assert len(ner_indices) == len(tokens)
        ner_tags = [self.ner_tags_list[index] for index in ner_indices]
        results = [(token, ner_tag) for token, ner_tag in zip(tokens, ner_tags)]

        return results

    def handle(self, data, context):
        """Entry point for default handler. It takes the data from the input request and returns
           the predicted outcome for the input.

        Args:
            data (list): The input data that needs to be made a prediction request on.
            context (Context): It is a JSON Object containing information pertaining to
                               the model artefacts parameters.

        Returns:
            list : Returns a list of dictionary with the predicted response.
        """
        logger.debug("HANDLE CALLED")
        # It can be used for pre or post processing if needed as additional request
        # information is available in context
        start_time = time.time()

        self.context = context
        metrics = self.context.metrics

        is_profiler_enabled = os.environ.get("ENABLE_TORCH_PROFILER", None)
        if is_profiler_enabled:
            output, _ = self._infer_with_profiler(data=data)
        else:
            input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags, tokens_count, tokens = self.preprocess(data)

            if not self._is_explain():
                output = self.inference(input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags)
                output = self.postprocess(output, tokens_count, tokens)
            else:
                output = self.explain_handle((), data)

        stop_time = time.time()
        metrics.add_time('HandlerTime', round(
            (stop_time - start_time) * 1000, 2), None, 'ms')
        return output
