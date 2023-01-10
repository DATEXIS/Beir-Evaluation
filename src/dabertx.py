from transformers import DABERTXModel, DABERTXConfig
from transformers import BertTokenizerFast
from transformers.models.dabertx import utils as dabertx_utils
import torch
from typing import List, Dict, Union
import numpy as np
from torch import Tensor


class DaBERTx:
    def __init__(self, device: str, model_name: str = "bert-base-uncased", tokenizer: str = 'bert-base-uncased',
                 batch_size: int = 32):
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = DABERTXModel(DABERTXConfig.from_pretrained(model_name))
        self.model.to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

    def encode(self, sentences, batch_size: int = 32, **kwargs):
        """ Returns a list of embeddings for the given sentences.
            Args:
                sentences (`List[str]`): List of sentences to encode
                batch_size (`int`): Batch size for the encoding

            Returns:
                `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
            """
        if not batch_size:
            batch_size = self.batch_size

        tokenized = self.tokenizer(sentences)
        tokenized = {"input_ids": tokenized["input_ids"],
                     "attention_mask": tokenized["attention_mask"],
                     "word_ids": [x.word_ids for x in tokenized.encodings]}

        sentence_embeddings = []
        with torch.no_grad():
            for i in range(0, len(tokenized["input_ids"]), batch_size):
                batch = {k: v[i:i + batch_size] for k, v in tokenized.items()}
                batch = dabertx_utils.sequence_to_word_batch_no_labels(batch, self.tokenizer)
                batch["input_ids"] = batch["input_ids"].to(self.device)
                batch["attention_mask"] = batch["attention_mask"].to(self.device)

                outputs = \
                    self.model(batch["input_ids"], batch["attention_mask"],
                               document_intervals=batch["document_intervals"],
                               return_dict=True)[
                        "last_hidden_state"][:, 0]
                sentence_embeddings.append(outputs)

            sentence_embeddings = torch.cat(sentence_embeddings).cpu()

        return sentence_embeddings

    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) \
            -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(self, corpus: Union[List[Dict[str, str]], Dict[str, List]], batch_size: int = 8, **kwargs) \
            -> Union[List[Tensor], np.ndarray, Tensor]:
        if type(corpus) is dict:
            sentences = [
                (corpus["title"][i] + self.sep + corpus["text"][i]).strip() if "title" in corpus else corpus["text"][
                    i].strip() for i in range(len(corpus['text']))]
        else:
            sentences = [(doc["title"] + self.sep + doc["text"]).strip() if "title" in doc else doc["text"].strip() for
                         doc in corpus]
        return self.encode(sentences, batch_size=batch_size, **kwargs)
