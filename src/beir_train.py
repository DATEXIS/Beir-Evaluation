from beir.retrieval.train import TrainRetriever
from sentence_transformers import losses
import os
import logging
from beir import util
from retrieve_dataset import DatasetRetriever


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeirTrain:
    def __init__(self, model,  dataset_retriever: DatasetRetriever, device: str, batch_size: int = 128):
        self.dataset_retriever = dataset_retriever
        self.batch_size = batch_size
        self.device = device
        self.model = model

    def train_model(self):
        corpus, queries, qrels = self.dataset_retriever.testset()
        dev_corpus, dev_queries, dev_qrels = self.dataset_retriever.devset()

        retriever = TrainRetriever(model=self.model, batch_size=self.batch_size)
        train_samples = retriever.load_train(corpus, queries, qrels)
        train_dataloader = retriever.prepare_train(train_samples, shuffle=True)

        train_loss = losses.MultipleNegativesRankingLoss(model=retriever.model, similarity_fct=util.dot_score)

        if dev_queries and dev_qrels and dev_corpus:
            ir_evaluator = retriever.load_ir_evaluator(dev_corpus, dev_queries, dev_qrels)
        else:
            ir_evaluator = retriever.load_dummy_evaluator()

        model_save_path = f'{self.dataset_retriever.output_dir}/output/' \
                          f'{self.model.model_name}-v1-{self.dataset_retriever.dataset}'

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
