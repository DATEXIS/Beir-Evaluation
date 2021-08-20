from beir_eval import BeirEval
from beir_train import BeirTrain
import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = os.getenv('DATASET', 'trec-covid')
    output_dir = os.getenv('OUTPUT_DIR', '/data')
    batch_size = int(os.getenv('BATCH_SIZE', '128'))
    bert_model = os.getenv('BERT_MODEL', 'bert-base-uncased')
    from_pretrained = os.getenv('FROM_PRETRAINED', '').lower() in ['true', '1']
    train = os.getenv('TRAIN', '').lower() in ['true', '1']
    logger.info(f'using device format {device}')
    if not train:
        logger.info(
            f'configs for evaluation: \n BERT_MODEL: {bert_model} \n DATASET: {dataset} \n OUTPUT_DIR: {output_dir} \n '
            f'BATCH_SIZE: {batch_size} \n FROM_PRETRAINED: {from_pretrained}')

        evaluation = BeirEval(bert_model, dataset, output_dir, device, batch_size)
        evaluation.evaluate_model(from_pretrained=from_pretrained)

    else:
        logger.info(
            f'configs for training: \n BERT_MODEL: {bert_model} \n DATASET: {dataset} \n OUTPUT_DIR: {output_dir} \n '
            f'BATCH_SIZE: {batch_size}')

        training = BeirTrain(bert_model, dataset, output_dir, device, batch_size)
        training.train_model()
