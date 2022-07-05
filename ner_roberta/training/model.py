import copy
import dvc.api
import torch.nn as nn
from transformers import RobertaForMaskedLM, RobertaConfig
import torch
from ner_roberta.training.metrics import cross_entropy_with_attention

class NerHead(nn.Module):
    def __init__(self, out_features, in_features=768):
        super(NerHead, self).__init__()
        self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, data):
        output = self.decoder(data)
        output = self.norm(output)
        return output


class RobertaNER(nn.Module):
    def __init__(self, base_roberta: RobertaForMaskedLM, ner_dict: dict,
                 pos_tags_count: int, model_config: dict, loss_function):
        super(RobertaNER, self).__init__()
        self.loss_function = loss_function
        self.pos_embeddings = nn.Embedding(pos_tags_count, model_config["POS_EMBEDDINGS_SIZE"])
        self.pos_embeddings_size = model_config["POS_EMBEDDINGS_SIZE"]
        self.default_sentence_len = model_config["DEFAULT_SENTENCE_LEN"]

        self.roberta = base_roberta
        self.ner_head = NerHead(len(ner_dict), in_features=768 + model_config["POS_EMBEDDINGS_SIZE"] + 1 + 1)
        del self.roberta.lm_head

    def forward(self, input_ids, token_type_ids, attention_mask, upcase_fracture, numbers_fracture, pos_tags,
                labels=None):
        outputs = self.roberta.roberta.forward(input_ids=input_ids, token_type_ids=token_type_ids,
                                               attention_mask=attention_mask)
        # batch * 512 * 768
        pos_vector = self.pos_embeddings(pos_tags)
        outputs = torch.cat(
            [outputs['last_hidden_state'], numbers_fracture.view(-1, self.default_sentence_len, 1), upcase_fracture.view(-1, self.default_sentence_len, 1),
             pos_vector], dim=2)
        outputs = self.ner_head(outputs)
        if labels is not None:
            # loss = nn.CrossEntropyLoss()( outputs.permute(0, 2, 1), labels )
            loss = self.loss_function(outputs.permute(0, 2, 1), labels, attention_mask)
            return loss, outputs.cpu()
        return (outputs,)


def build_model_from_train_checkpoint(ner_tags_dict: dict, pos_tags_count: int, model_config: dict, model_file_path: str):
    roberta_base = RobertaForMaskedLM(RobertaConfig(
        vocab_size=50265,
        max_position_embeddings=514,
        type_vocab_size=1
    ))
    ner_roberta = RobertaNER(roberta_base, ner_tags_dict, pos_tags_count, model_config, cross_entropy_with_attention)
    ner_roberta.load_state_dict(torch.load(model_file_path))

    return ner_roberta