import pandas as pd
import torch
import tqdm
from transformers.file_utils import PaddingStrategy
from transformers import RobertaTokenizer
from ner_roberta.training.config import MainConfig
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')



class NerDataset:
    def __init__(self, input_df : pd.DataFrame, tags_dict : dict, pos_tags_dict : dict, tokenizer : RobertaTokenizer, params: dict ):

        self.tokenizer_output = []
        self.ner_tags = []
        self.tokenizer = tokenizer
        self.device = torch.device(params["DATA"]["DEVICE"])
        self.uppercase_fracture = []
        self.numbers_fracture = []
        self.pos_tags = []

        DEFAULT_SENTENCE_LEN = params["MODEL"]["DEFAULT_SENTENCE_LEN"]

        for index, row in tqdm.tqdm(input_df.iterrows(), total=input_df.shape[0]):
            sentence = row['Sentence']

            tokenizer_output = tokenizer(sentence, padding = PaddingStrategy.MAX_LENGTH, truncation = True, max_length = DEFAULT_SENTENCE_LEN, return_token_type_ids = True, return_attention_mask = True, return_special_tokens_mask=True )
            tags_list = row['Tag'].replace('[', '').replace(']', '').replace("'", "")
            tags_list = [word.strip() for word in tags_list.split(',')]
            tags_list = tags_list[ : DEFAULT_SENTENCE_LEN - 1]

            tags_list = [params["tags"]["NER"]["CLS_TAG"]] + tags_list + [params["tags"]["NER"]["CLS_TAG"]] * (DEFAULT_SENTENCE_LEN - len(tags_list) - 1)
            tags_list = [ tags_dict.get(tag, tags_dict[params["tags"]["NER"]["UNK_TAG"]]) for tag in tags_list ]
            self.ner_tags.append(tags_list)
            self.tokenizer_output.append(tokenizer_output)

            tokens = nltk.tokenize.word_tokenize(sentence)
            pos_tags = [pos_tags_dict.get(item[1], pos_tags_dict[params["tags"]["POS"]["UNK_POS_TAG"]])  for item in nltk.pos_tag(tokens)]

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
                upcase_fracture.append( token_upcase_count / len(token) )
                numbers_fracture.append( numbers_count / len(token) )
            upcase_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(upcase_fracture))
            numbers_fracture += [0] * (DEFAULT_SENTENCE_LEN - len(numbers_fracture))
            pos_tags += [pos_tags_dict[params["tags"]["POS"]["PAD_POS_TAG"]]] * (DEFAULT_SENTENCE_LEN - len(pos_tags))
            self.uppercase_fracture.append(upcase_fracture)
            self.numbers_fracture.append(numbers_fracture)
            self.pos_tags.append(pos_tags)


    def __len__(self):
        return len(self.tokenizer_output)

    def __getitem__(self, index):
        output = self.tokenizer_output[index]
        input_ids = output['input_ids']
        token_type_ids = output['token_type_ids']
        # special_tokens_mask = output['special_tokens_mask']
        attention_mask = output['attention_mask']

        input_ids = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=self.device)
        # special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=self.device)

        ner_tags = self.ner_tags[index]
        ner_tags = torch.tensor(ner_tags, dtype=torch.long, device=self.device)

        upcase_fracture = torch.tensor(self.uppercase_fracture[index], dtype=torch.float, device=self.device)
        numbers_fracture = torch.tensor(self.numbers_fracture[index], dtype=torch.float, device=self.device)
        pos_tags = torch.tensor(self.pos_tags[index], dtype=torch.long, device=self.device)
        return {
            "input_ids" : input_ids, "token_type_ids" : token_type_ids, "attention_mask" : attention_mask, "labels" : ner_tags,
            "upcase_fracture" : upcase_fracture, "numbers_fracture" : numbers_fracture, "pos_tags" : pos_tags
        }

