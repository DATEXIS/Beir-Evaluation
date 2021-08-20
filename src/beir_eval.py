from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging
from transformers import BertTokenizerFast
import os
from biencoder import BiEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeirEval:
    def __init__(self, bert_model: str, dataset: str, output_dir: str, device: str, batch_size: int = 128):
        self.dataset = dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.bert_model = bert_model

    def evaluate_model(self, from_pretrained: bool = True):
        logger.info(f'starting evaluation dataset {self.dataset}')
        output_dir_data = f'{self.output_dir}/{self.dataset}'
        if os.path.isdir(output_dir_data):
            logger.info(f'dataset {self.dataset} already downloaded')
            data_path = f'{output_dir_data}/{self.dataset}'
        else:
            url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip'
            data_path = util.download_and_unzip(url, output_dir_data)

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

        tokenizer = BertTokenizerFast.from_pretrained(self.bert_model, do_lower_case=('uncased' in self.bert_model))
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ent]']})

        if from_pretrained:
            biencoder = BiEncoder.from_pretrained(model_path=self.output_dir, tokenizer=tokenizer, device=self.device,
                                                bert_model=self.bert_model)
        else:
            biencoder = BiEncoder(device=self.device, tokenizer=tokenizer, bert_model=self.bert_model)
        biencoder.eval()

        logger.info(f'loading model with batch size {self.batch_size}')
        model = DRES(biencoder, batch_size=self.batch_size)

        retriever = EvaluateRetrieval(model, score_function='cos_sim')
        results = retriever.retrieve(corpus, queries)

        ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        logger.info(f'results: \n ndcg: {ndcg} \n map: {map} \n recall: {recall} \n precision: {precision}')



