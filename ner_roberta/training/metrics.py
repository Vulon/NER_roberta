import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score
import copy

def cross_entropy_with_attention(a, b, attention_mask):
    a = torch.nn.LogSoftmax(dim=1)(a)
    loss = nn.NLLLoss(reduction='none')(a, b)
    loss = loss * attention_mask
    return loss.mean()


def create_compute_metrics_function(ner_tags_dict: dict, ner_cls_tag):
    ner_tags_dict = copy.deepcopy(ner_tags_dict)

    def compute_metrics(evalPrediction):
        prediction = np.argmax(evalPrediction.predictions, axis=2).flatten()
        y = evalPrediction.label_ids.flatten()
        mask = (y != ner_tags_dict[ner_cls_tag])
        prediction = prediction[mask]
        y = y[mask]

        recall = recall_score(prediction, y, average='macro', labels=list(ner_tags_dict.values()))
        precision = precision_score(prediction, y, average='macro', labels=list(ner_tags_dict.values()))
        # f1_micro = f1_score(prediction, y, average='micro', labels=list(ner_tags_dict.values()))
        f1_macro = f1_score(prediction, y, average='macro', labels=list(ner_tags_dict.values()))
        return {"recall": recall, "precision": precision, "f1_macro": f1_macro}

    return compute_metrics