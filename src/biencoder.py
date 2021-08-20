import torch
from torch import nn, Tensor
from transformers import BertModel, BatchEncoding, BertTokenizerFast
import os
from typing import List, Dict
from tqdm.autonotebook import trange
import numpy as np


class Encoder(nn.Module):

    def __init__(self, len_of_token_embeddings: int, device: str, bert_model: str):
        super(Encoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        for param in list(self.bert.embeddings.parameters()):
            param.requires_grad = False
        self.bert.resize_token_embeddings(len_of_token_embeddings)

    # def forward(self, token_ids: Tensor) -> Tensor:
    #     hidden_states, cls_tokens = self.bert(token_ids, return_dict=False)
    #     return cls_tokens

    def forward(self, token_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden_states, cls_tokens = self.bert(token_ids, attention_mask=attention_mask, return_dict=False)
        return cls_tokens


class BiEncoder:

    def __init__(self, device: str, tokenizer: BertTokenizerFast,
                 freeze_embeddings: bool = True, bert_model: str = 'bert-base-uncased'):
        self.device: str = device
        self.tokenizer: BertTokenizerFast = tokenizer
        self.bert_model = bert_model
        self.max_length = None
        self.ent_token = '[ent]'
        tokenizer_len = len(self.tokenizer)
        self.encoder_mention: Encoder = Encoder(len_of_token_embeddings=tokenizer_len, device=self.device,
                                                bert_model=self.bert_model).to(self.device)
        self.encoder_concept: Encoder = Encoder(len_of_token_embeddings=tokenizer_len, device=self.device,
                                                bert_model=self.bert_model).to(self.device)

    @classmethod
    def from_pretrained(cls, model_path: str, tokenizer: BertTokenizerFast,
                        device: str, bert_model: str) -> 'BiEncoder':
        biencoder = cls(device=device, tokenizer=tokenizer, bert_model=bert_model)
        mention_encoder_path = os.path.join(model_path, 'encoder_mention.statedict')
        concept_encoder_path = os.path.join(model_path, 'encoder_mention.statedict')
        # concept_encoder_path = os.path.join(model_path, 'encoder_concept.statedict')
        biencoder.encoder_mention.load_state_dict(torch.load(mention_encoder_path, map_location=device))
        biencoder.encoder_concept.load_state_dict(torch.load(concept_encoder_path, map_location=device))
        return biencoder

    def eval(self):
        self.encoder_mention.eval()
        self.encoder_concept.eval()

    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:
        # https://github.com/UKPLab/beir/blob/b4346f88e343b0886de6c03a90a1d71948bbd3c3/beir/retrieval/models/dpr.py#L21
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.tokenizer(queries[start_idx:start_idx + batch_size], truncation=True, padding=True,
                                         return_tensors='pt')
                cls_tokens = self.encoder_concept(encoded['input_ids'].cuda(),
                                                  attention_mask=encoded['attention_mask'].cuda())
                query_embeddings += cls_tokens.cpu().squeeze()

        return torch.stack(query_embeddings)

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs) -> np.ndarray:
        # https://github.com/UKPLab/beir/blob/b4346f88e343b0886de6c03a90a1d71948bbd3c3/beir/retrieval/models/dpr.py#L31
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                titles = [row['title'] for row in corpus[start_idx:start_idx + batch_size]]
                texts = [row['text'] for row in corpus[start_idx:start_idx + batch_size]]
                encoded = self.tokenizer(titles, texts, truncation='longest_first', padding=True,
                                         return_tensors='pt')
                cls_tokens = self.encoder_mention(encoded['input_ids'].cuda(),
                                                  attention_mask=encoded['attention_mask'].cuda())
                corpus_embeddings += cls_tokens.cpu().squeeze()

        return torch.stack(corpus_embeddings)
