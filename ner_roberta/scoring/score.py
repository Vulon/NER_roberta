from transformers import RobertaForMaskedLM, RobertaConfig, RobertaTokenizer
import json, os
from transformers.file_utils import PaddingStrategy
import torch.nn as nn
import torch
import nltk



class RobertaNerScore(nn.Module):

    class NerHead(nn.Module):
        def __init__(self, out_features, in_features=768):
            super(RobertaNerScore.NerHead, self).__init__()
            self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.norm = nn.LayerNorm(out_features)

        def forward(self, data):
            output = self.decoder(data)
            output = self.norm(output)
            return output
    def __init__(
            self,
            pos_tags_count: int,
            pos_embeddings_size: int,
            ner_tags_count: int,
            default_sentence_len: int,
            **kwargs
        ):
        super(RobertaNerScore, self).__init__()
        self.roberta = RobertaForMaskedLM(RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            type_vocab_size=1
        ))

        self.pos_embeddings = nn.Embedding(pos_tags_count, pos_embeddings_size)
        self.ner_head = RobertaNerScore_.NerHead(ner_tags_count, in_features=768 + pos_embeddings_size + 1 + 1)
        del self.roberta.lm_head
        self.default_sentence_len = default_sentence_len

    def forward(self, input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags, *args, **kwargs):
        outputs = self.roberta.roberta.forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                               attention_mask=attention_mask)
        pos_vector = self.pos_embeddings(pos_tags)
        outputs = torch.cat(
            [outputs['last_hidden_state'],
             numbers_fracture.view(-1, self.default_sentence_len, 1),
             upcase_fracture.view(-1, self.default_sentence_len, 1),
             pos_vector], dim=2
        )
        outputs = self.ner_head(outputs)
        return outputs

class RobertaNerScore_(nn.Module):

    class NerHead(nn.Module):
        def __init__(self, out_features, in_features=768):
            super(RobertaNerScore_.NerHead, self).__init__()
            self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.norm = nn.LayerNorm(out_features)

        def forward(self, data):
            output = self.decoder(data)
            output = self.norm(output)
            return output

    def __init__(
            self, base_roberta, ner_tags_dict, pos_tags_dict,
            tokenizer, package_dir, mini_config, ner_tags_list
    ):
        super(RobertaNerScore_, self).__init__()
        pos_embeddings_size = mini_config["config.MODEL.POS_EMBEDDINGS_SIZE"]
        self.pos_embeddings = nn.Embedding(len(pos_tags_dict), pos_embeddings_size)
        self.roberta = base_roberta
        self.ner_head = RobertaNerScore_.NerHead(len(ner_tags_dict), in_features=768 + pos_embeddings_size + 1 + 1)
        del self.roberta.lm_head
        self.ner_tags_dict = ner_tags_dict
        self.pos_tags_dict = pos_tags_dict
        self.tokenizer = tokenizer
        self.package_dir = package_dir
        self.mini_config = mini_config
        self.ner_tags_list = ner_tags_list

    @classmethod
    def from_path(cls, package_dir):
        with open(os.path.join(package_dir, "config.json"), 'r') as file:
            mini_config = json.load(file)

        roberta_base = RobertaForMaskedLM(RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            type_vocab_size=1
        ))

        with open(os.path.join(package_dir, "ner_tags_dict.json"), 'r') as file:
            ner_tags_dict = json.load(file)
        with open(os.path.join(package_dir, "pos_tags_dict.json"), 'r') as file:
            pos_tags_dict = json.load(file)
        ner_tags_list = sorted(ner_tags_dict, key=lambda x: ner_tags_dict[x])

        ner_roberta = cls(
            roberta_base,
            ner_tags_dict, pos_tags_dict,
            RobertaTokenizer.from_pretrained(os.path.join(package_dir, "tokenizer")),
            package_dir,
            mini_config,
            ner_tags_list
        )
        ner_roberta.load_state_dict(torch.load(os.path.join(package_dir, "model.pt")))

        return ner_roberta

    def _prepare_data(self, sentence):
        os.environ['NLTK_DATA'] = os.path.join(self.package_dir, "NLTK")
        mini_config = self.mini_config
        tokens = nltk.tokenize.word_tokenize(sentence)
        pos_tags_dict = self.pos_tags_dict
        DEFAULT_SENTENCE_LEN = mini_config["config.MODEL.DEFAULT_SENTENCE_LEN"]

        pos_tags = [pos_tags_dict.get(item[1], pos_tags_dict[mini_config["config.POS.UNK_POS_TAG"]]) for item in
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
        pos_tags += [pos_tags_dict[mini_config["config.POS.PAD_POS_TAG"]]] * (DEFAULT_SENTENCE_LEN - len(pos_tags))
        tokenizer_output = self.tokenizer(sentence, padding=PaddingStrategy.MAX_LENGTH, truncation=True,
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

    def predict(self, sentence, **kwargs):
        input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags, tokens_count, tokens = self._prepare_data(
            sentence)
        outputs = self.roberta.roberta.forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                               attention_mask=attention_mask)
        pos_vector = self.pos_embeddings(pos_tags)
        outputs = torch.cat(
            [outputs['last_hidden_state'],
             numbers_fracture.view(-1, self.mini_config["config.MODEL.DEFAULT_SENTENCE_LEN"], 1),
             upcase_fracture.view(-1, self.mini_config["config.MODEL.DEFAULT_SENTENCE_LEN"], 1),
             pos_vector], dim=2
        )
        outputs = self.ner_head(outputs)
        ner_indices = outputs.max(dim=2).indices.flatten().tolist()
        ner_indices = ner_indices[1: tokens_count + 1]
        assert len(ner_indices) == len(tokens)
        ner_tags = [self.ner_tags_list[index] for index in ner_indices]
        results = [(token, ner_tag) for token, ner_tag in zip(tokens, ner_tags)]
        return results
