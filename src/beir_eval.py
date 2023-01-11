from retrieve_dataset import DatasetRetriever
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeirEval:
    def __init__(self, bert_model: str, dataset_retriever: DatasetRetriever, device: str, batch_size: int = 128):
        self.dataset_retriever = dataset_retriever
        self.batch_size = batch_size
        self.device = device
        self.bert_model = bert_model

    def evaluate_model(self):
        corpus, queries, qrels = self.dataset_retriever.testset()

        logger.info(f'loading model with batch size {self.batch_size}')
        model = DRES(self.model, batch_size=self.batch_size)

        retriever = EvaluateRetrieval(model, score_function='cos_sim')
        results = retriever.retrieve(corpus, queries)

        ndcg, map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

        logger.info(f'results: \n ndcg: {ndcg} \n map: {map} \n recall: {recall} \n precision: {precision}')



