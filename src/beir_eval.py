from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
from transformers import BertModel, BertTokenizerFast
import os
import torch
from torch import nn, Tensor

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)


class DocumentEncoder(nn.Module):

    def __init__(self, len_of_token_embeddings: int, device: str, bert_model: str):
        super(DocumentEncoder, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model).to(device)
        self.bert.resize_token_embeddings(len_of_token_embeddings)

    def forward(self, token_ids: Tensor, attention_masks: Tensor) -> Tensor:
        hidden_states, cls_tokens = self.bert(token_ids, attention_mask=attention_masks, return_dict=False)
        return cls_tokens

    @classmethod
    def from_pretrained(cls, path_to_statedict: str, tokenizer: BertTokenizerFast, device: str, bert_model: str) -> 'BiEncoder':
        document_encoder = cls(len_of_token_embeddings=len(tokenizer), device=device, bert_model=bert_model)
        document_encoder.load_state_dict(torch.load(path_to_statedict, map_location=device))
        return document_encoder


class BeirEval:
    def __init__(self, bert_model, dataset, output_dir, device, batch_size=128):
        self.dataset = dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.bert_model = bert_model

    def evaluate_model(self):
        logger.info(f'starting evaluation dataset {self.dataset}')
        output_dir_data = f'{self.output_dir}/{self.dataset}'
        if os.path.isdir(output_dir_data):
            logger.info(f'dataset {self.dataset} already downloaded')
            data_path = f'{output_dir_data}/{self.dataset}'
        else:
            url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip'
            data_path = util.download_and_unzip(url, output_dir_data)

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        tokenizer = BertTokenizerFast.from_pretrained(self.bert_model, do_lower_case=('uncased' in bert_model))
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ent]']})


        encoder_path = os.path.join(self.output_dir, 'encoder_mention.statedict')
        document_encoder = DocumentEncoder.from_pretrained(path_to_statedict=encoder_path, tokenizer=tokenizer,
                                               device=self.device, bert_model=self.bert_model)
        document_encoder.eval()

        logger.info(f'loading model with batch size {self.batch_size}')
        model = DRES(document_encoder, batch_size=self.batch_size)

        retriever = EvaluateRetrieval(model, score_function='cos_sim')
        results = retriever.retrieve(corpus, queries)

        ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        logger.info(f'results: \n ndcg: {ndcg} \n map: {map} \n recall: {recall} \n precision: {precision}')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = os.getenv('DATASET', 'trec-covid')
    output_dir = os.getenv('OUTPUT_DIR', '/data')
    batch_size = int(os.getenv('BATCH_SIZE', '128'))
    bert_model = os.getenv('BERT_MODEL', 'bert-base-uncased')
    logger.info(f'using device format {device}')
    logger.info(f'configs for evaluation: \n BERT_MODEL: {bert_model} \n DATASET: {dataset} \n OUTPUT_DIR: {output_dir} \n BATCH_SIZE: {batch_size}')


    eval = BeirEval(bert_model, dataset, output_dir, device, batch_size)
    eval.evaluate_model()
