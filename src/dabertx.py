from transformers import DABERTXModel, DABERTXConfig
from transformers import BertTokenizerFast
from transformers.models.dabertx import utils as dabertx_utils
import torch


class DaBERTx:
    def __init__(self, device: str, model_name: str = "bert-base-uncased", tokenizer: str = 'bert-base-uncased', batch_size: int = 32):
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = DABERTXModel(DABERTXConfig.from_pretrained(model_name))
        self.model.to(device)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)

    def encode(self, sentences, **kwargs):
        """ Returns a list of embeddings for the given sentences.
            Args:
                sentences (`List[str]`): List of sentences to encode
                batch_size (`int`): Batch size for the encoding

            Returns:
                `List[np.ndarray]` or `List[tensor]`: List of embeddings for the given sentences
            """

        tokenized = self.tokenizer(sentences)
        tokenized = {"input_ids": tokenized["input_ids"],
                     "attention_mask": tokenized["attention_mask"],
                     "word_ids": [x.word_ids for x in tokenized.encodings]}

        sentence_embeddings = []
        with torch.no_grad():
            for i in range(0, len(tokenized["input_ids"]), self.batch_size):
                batch = {k: v[i:i + self.batch_size] for k, v in tokenized.items()}
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
