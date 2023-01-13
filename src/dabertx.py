import logging
from transformers import DABERTXModel, DABERTXConfig
from transformers import BertTokenizerFast
from transformers.models.dabertx import utils as dabertx_utils
import torch
from typing import List, Dict, Union, Tuple, Iterable, Type, Callable
import numpy as np
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.readers.InputExample import InputExample
from tqdm.autonotebook import trange
import transformers
import os
from transformers import DataCollatorForLanguageModeling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# class DABERTXCollator:
#     def __init__(self, tokenizer, collator):
#         self.tokenizer = tokenizer
#         self.collator = collator
#
#     def __call__(self, examples):
#         # flatten into a single dict
#         logger.info(f"EXAMPLES:\n{examples}")
#         if isinstance(examples, InputExample):
#             new_examples = []
#             dabertx_utils.sequence_to_word_batch(new_examples, self.tokenizer, self.collator)
#
#         else:
#             new_examples = {k: [v] for k, v in examples[0].items()}
#             for example in examples[1:]:
#                 for k in example.keys():
#                     new_examples[k].append(example[k])
#
#             return dabertx_utils.sequence_to_word_batch(new_examples, self.tokenizer, self.collator)


def batch_to_device(batch, target_device: str):
    """
    send a pytorch batch to a device (CPU/GPU)
    """
    for key in batch:
        key = key.long()
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch


class DaBERTx:
    def __init__(self, device: str, model_name: str = "bert-base-uncased", tokenizer: str = 'bert-base-uncased',
                 batch_size: int = 32):
        self.batch_size = batch_size
        self.model_name = model_name
        self.model = DABERTXModel(DABERTXConfig.from_pretrained(model_name))
        self.model.to(device)
        self.device = device
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

    def smart_batching_collate(self, batch):
        """
            Transforms a batch from a SmartBatchingDataset to a batch of tensors for the model
            Here, batch is a list of tuples: [(tokens, label), ...]
            :param batch:
                a batch from a SmartBatchingDataset
            :return:
                a batch of tensors for the model
            """
        num_texts = len(batch[0].texts)
        texts = [[] for _ in range(num_texts)]
        labels = []

        for example in batch:
            for idx, text in enumerate(example.texts):
                texts[idx].append(text)

            labels.append(example.label)

        labels = torch.tensor(labels)

        sentence_features = []
        for idx in range(num_texts):
            tokenized = self.encode(texts[idx])
            sentence_features.append(tokenized)

        return sentence_features, labels

    def fit(self,
            train_objectives: Iterable[Tuple[DataLoader, nn.Module]],
            evaluator: SentenceEvaluator = None,
            epochs: int = 1,
            steps_per_epoch=None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW,
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            decay_steps=100_000,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            checkpoint_path: str = '/tmp/checkpoint',
            checkpoint_save_steps: int = 500,
            checkpoint_save_total_limit: int = 0,
            **kwargs
            ):

        # TODO:
        checkpoint_saves = 0
        dataloaders = [dataloader for dataloader, _ in train_objectives]

        for dataloader in dataloaders:
            # TODO:
            # collator = DABERTXCollator(self.tokenizer, DataCollatorForLanguageModeling(self.tokenizer, mlm=True,
            #                                                                            pad_to_multiple_of=64))
            dataloader.collate_fn = self.smart_batching_collate
            # dataloader.collate_fn = collator

        loss_models = [loss for _, loss in train_objectives]

        self.best_score = -9999999

        if steps_per_epoch is None or steps_per_epoch == 0:
            steps_per_epoch = min([len(dataloader) for dataloader in dataloaders])

        num_train_steps = int(steps_per_epoch * epochs)

        optimizers = []
        schedulers = []
        for loss_model in loss_models:
            param_optimizer = list(loss_model.named_parameters())

            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': weight_decay},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
            scheduler_obj = transformers.get_polynomial_decay_schedule_with_warmup(optimizer, warmup_steps,
                                                                                   num_train_steps)
            optimizers.append(optimizer)
            schedulers.append(scheduler_obj)
            loss_model.to(self.device)

        global_step = 0
        data_iterators = [iter(dataloader) for dataloader in dataloaders]
        num_train_objectives = len(train_objectives)
        skip_scheduler = False
        for epoch in trange(epochs, desc="Epoch"):
            training_steps = 0

            for loss_model in loss_models:
                loss_model.zero_grad()
                loss_model.train()

            for _ in trange(steps_per_epoch, desc="Iteration", smoothing=0.05):
                for train_idx in range(num_train_objectives):
                    loss_model = loss_models[train_idx]
                    optimizer = optimizers[train_idx]
                    scheduler = schedulers[train_idx]
                    data_iterator = data_iterators[train_idx]

                    try:
                        data = next(data_iterator)
                    except StopIteration:
                        data_iterator = iter(dataloaders[train_idx])
                        data_iterators[train_idx] = data_iterator
                        data = next(data_iterator)

                    features, labels = data
                    labels = labels.to(self.device)
                    features = list(map(lambda batch: batch_to_device(batch, self.device), features))

                    # if use_amp:
                    #     with autocast():
                    #         loss_value = loss_model(features, labels)
                    #
                    #     scale_before_step = scaler.get_scale()
                    #     scaler.scale(loss_value).backward()
                    #     scaler.unscale_(optimizer)
                    #     torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    #     scaler.step(optimizer)
                    #     scaler.update()
                    #
                    #     skip_scheduler = scaler.get_scale() != scale_before_step
                    # else:
                    loss_value = loss_model(features, labels)
                    loss_value.backward()
                    torch.nn.utils.clip_grad_norm_(loss_model.parameters(), max_grad_norm)
                    optimizer.step()

                    optimizer.zero_grad()

                    if not skip_scheduler:
                        scheduler.step()

                training_steps += 1
                global_step += 1

                if evaluation_steps > 0 and training_steps % evaluation_steps == 0:
                    score = evaluator(self, output_path=output_path, epoch=epoch, steps=training_steps)
                    if callback is not None:
                        callback(score, epoch, training_steps)
                    if score > self.best_score:
                        self.best_score = score
                        if save_best_model:
                            self.save(output_path)
                    for loss_model in loss_models:
                        loss_model.zero_grad()
                        loss_model.train()

                if checkpoint_path is not None and checkpoint_save_steps is not None and checkpoint_save_steps > 0 \
                        and global_step % checkpoint_save_steps == 0 and checkpoint_save_total_limit > checkpoint_saves:
                    self.model.save(os.path.join(checkpoint_path, str(global_step)))
