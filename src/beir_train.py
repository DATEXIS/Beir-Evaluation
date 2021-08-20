from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.train import TrainRetriever
import os
import logging
from beir import util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeirTrain:
    def __init__(self, bert_model: str, dataset: str, output_dir: str, device: str, batch_size: int = 128):
        self.dataset = dataset
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.device = device
        self.bert_model = bert_model

    def train_model(self):
        logger.info(f'starting evaluation dataset {self.dataset}')
        output_dir_data = f'{self.output_dir}/{self.dataset}'
        if os.path.isdir(output_dir_data):
            logger.info(f'dataset {self.dataset} already downloaded')
            data_path = f'{output_dir_data}/{self.dataset}'
        else:
            url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset}.zip'
            data_path = util.download_and_unzip(url, output_dir_data)

        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
        dev_corpus, dev_queries, dev_qrels = GenericDataLoader(data_folder=data_path).load(split="dev")
        # todo: provide model, in example it is used SentenceTransformer, is this correct?
        # https://github.com/UKPLab/beir/blob/933b349bf300718cd6a2d285c51fe78f48fdec85/examples/retrieval/training/train_sbert.py#L46
        model = ''

        retriever = TrainRetriever(model=model, batch_size=self.batch_size)
        train_samples = retriever.load_train(corpus, queries, qrels)
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

        # todo: uses loss from sentence_transformer, is this correcT?
        # train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model)
        train_loss = ''

        if dev_queries and dev_qrels and dev_corpus:
            ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
        else:
            ir_evaluator = retriever.load_dummy_evaluator()

        model_save_path = f'{self.output_dir}/output/{self.model_name}-v1-{self.dataset}'

        os.makedirs(model_save_path, exist_ok=True)

        #### Configure Train params
        num_epochs = 1
        evaluation_steps = 5000
        warmup_steps = int(len(train_samples) * num_epochs / retriever.batch_size * 0.1)

        retriever.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=ir_evaluator,
                      epochs=num_epochs,
                      output_path=model_save_path,
                      warmup_steps=warmup_steps,
                      evaluation_steps=evaluation_steps,
                      use_amp=True)