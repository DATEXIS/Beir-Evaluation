from beir_eval import BeirEval
from beir_train import BeirTrain
import torch
import logging
import argparse
from retrieve_dataset import DatasetRetriever
from dabertx import DaBERTx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Evaluating model with MTEB Benchmark")
    parser.add_argument('--model_or_path', type=str, default="bert-base-uncased",
                        help="Model name or path to be evaluated")
    parser.add_argument('--tokenizer', type=str, default="bert-base-uncased", help="Tokenizer to be used")
    parser.add_argument('--output_path', type=str, default="/data",
                        help="Output path for evaluation results,"
                             " timestamp will be added")
    parser.add_argument('--dataset', nargs='+', type=str, default='trec-covid',
                        help="Dataset to use for train and/or evaluation")
    parser.add_argument('--train', type=bool, default=False, help="Train model")
    parser.add_argument('--eval', type=bool, default=False, help="Evaluate model")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size ")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'using device format {device}')

    dataset_retriever = DatasetRetriever(args.dataset, args.output_path)
    dabertx = DaBERTx(device=device, model_name=args.model_or_path, tokenizer=args.tokenizer, batch_size=args.batch_size)

    if args.train:
        logger.info(
            f'configs for training: \n BERT_MODEL: {args.model_or_path} \n DATASET: {args.dataset} \n OUTPUT_DIR: '
            f'{args.output_path} \n BATCH_SIZE: {args.batch_size}')

        training = BeirTrain(model=dabertx, dataset_retriever=dataset_retriever, device=device,
                             batch_size=args.batch_size)
        training.train_model()

    if args.eval:
        logger.info(
            f'configs for evaluation: \n BERT_MODEL: {args.model_or_path} \n DATASET: {args.dataset} \n '
            f'OUTPUT_DIR: {args.output_path} \n BATCH_SIZE: {args.batch_size}')

        evaluation = BeirEval(model=dabertx, dataset_retriever=dataset_retriever, device=device,
                              batch_size=args.batch_size)
        evaluation.evaluate_model()
