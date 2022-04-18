from transformers import RobertaForMaskedLM, RobertaConfig
import torch.nn as nn
import torch


class RobertaNerScore(nn.Module):

    class NerHead(nn.Module):
        def __init__(self, out_features: int, in_features=768):
            """
            Named Entity Recognition head, replaces the Language model head.
            Takes torch.Tensor as an input from roberta base and predicts NER tags for each input token.
            :param out_features: the amount of classes to predict (length of the NER dictionary)
            :param in_features: the size of the roberta base output vector
            """
            super(RobertaNerScore.NerHead, self).__init__()
            self.decoder = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
            self.norm = nn.LayerNorm(out_features)

        def forward(self, data: torch.Tensor) -> torch.Tensor:
            """
            Takes predictions from roberta base as an input.
            Predicts probabilities distribution among the NER tags for each input token.
            :param data: output from roberta base
            :return:
            """
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
        """
        RobertaNerScore - simplified ner_roberta.training.model.RobertaNer class for scoring the model.
        It takes BPE tokens and additional character level information to predict NER tags.
        This model uses RoBERTa base model for named entity recognition, pretrained on English language
        using a masked language modeling (MLM) objective . https://huggingface.co/roberta-base .
        :param pos_tags_count: POS tags dictionary size
        :param pos_embeddings_size: embeddings size for POS tags
        :param ner_tags_count: NER tags dictionary size
        :param default_sentence_len: Maximum input tokens count.
        :param kwargs:
        """
        super(RobertaNerScore, self).__init__()

        self.roberta = RobertaForMaskedLM(RobertaConfig(
            vocab_size=50265,
            max_position_embeddings=514,
            type_vocab_size=1
        ))

        self.pos_embeddings = nn.Embedding(pos_tags_count, pos_embeddings_size)
        self.ner_head = RobertaNerScore.NerHead(ner_tags_count, in_features=768 + pos_embeddings_size + 1 + 1)
        del self.roberta.lm_head
        self.default_sentence_len = default_sentence_len

    def forward(
            self,
            input_ids: torch.Tensor,
            token_type_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            upcase_fracture: torch.Tensor,
            numbers_fracture: torch.Tensor,
            pos_tags: torch.Tensor,
            *args, **kwargs
            ) -> torch.Tensor:
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
