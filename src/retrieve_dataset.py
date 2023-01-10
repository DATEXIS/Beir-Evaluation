import logging
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetRetriever:
    def __init__(self, dataset: str, output_dir: str):
        self.dataset = dataset
        self.output_dir = output_dir
        self.output_dir_data = f'{self.output_dir}/{self.dataset}'
        if os.path.isdir(self.output_dir_data):
            logger.info(f'self.dataset {self.dataset} already downloaded')
            self.data_path = f'{self.output_dir_data}/{self.dataset}'
        else:
            url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip'
            self.data_path = util.download_and_unzip(url, self.output_dir_data)

    def testset(self):
        corpus, queries, qrels = GenericDataLoader(data_folder=self.data_path).load(split="test")
        return corpus, queries, qrels

    def devset(self):
        if not os.path.exists(f'{self.data_path}/qrels/dev.tsv'):
            dev_corpus = dev_queries = dev_qrels = None
        else:
            dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_folder=self.data_path).load(split="dev")
        return dev_corpus, dev_queries, dev_qrels
