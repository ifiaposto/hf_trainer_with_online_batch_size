#MIT License
#
#Copyright (c) 2025 Ifigeneia Apostolopoulou
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

import random
from functools import partial
import torch
from datasets import Dataset

from transformers import (TrainingArguments, AutoModelForSequenceClassification,
                          AutoTokenizer, HfArgumentParser)

from src.adaptive_batch_trainer import AdaptiveBatchSizeTrainer
"""" 
Test code for adaptive batch size training based on huggingface's trainer and sampler.

Sample run commands are:

drop last True (all batches have currently set batch size ):

single gpu:

python -m torch.distributed.run --nproc-per-node=1 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last
    
multi gpu:

python -m torch.distributed.run --nproc-per-node=2 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last
    
drop last False (last batch might have size less than the currently set batch size ):

single gpu:

python -m torch.distributed.run --nproc-per-node=1 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last False
    
multi gpu:

python -m torch.distributed.run --nproc-per-node=2 --master_port=29619 -m test_replay_dataloader \
    --output_dir ./results \
    --logging_dir ./logs \
    --logging_steps 1 \
    --save_strategy no \
    --num_train_epochs 10 \
    --per_device_train_batch_size 1 \
    --max_steps -1 \
    --dataloader_drop_last False
    
"""

# Tokenizer and collate function
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def collate_fn(batch):
    texts = [item["text"] for item in batch]
    labels = [item["label"] for item in batch]
    indices = [item["index"] for item in batch]
    encodings = tokenizer(texts,
                          padding=True,
                          truncation=True,
                          return_tensors="pt")
    encodings["labels"] = torch.tensor(labels)
    encodings["indices"] = torch.tensor(indices)
    return encodings


# Initialize dummy dataset and model
num_train_examples = 10
dataset = dataset = Dataset.from_dict({
    "text": [f"Sample text {i + 1}" for i in range(num_train_examples)],
    "label": [random.randint(0, 1) for _ in range(num_train_examples)],
    "index": [i for i in range(num_train_examples)]
})

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
                                                           num_labels=2)

parser = HfArgumentParser(TrainingArguments)
training_args = parser.parse_args_into_dataclasses()[0]


#   < ------- Write your own batch size scheduler ------- >
def custom_batch_size_scheduler(step: int,
                                batch_size: int,
                                interval=5,
                                increment=1):
    """
        step: current optimization step to be provided by the trainer.
        batch_size: current optimization step  to be provided by the trainer.
    """

    if step % interval == 0 and step > 0:
        return batch_size + increment
    return batch_size


# Extended trainer using the adaptive batch sampler.
trainer = AdaptiveBatchSizeTrainer(
    batch_size_scheduler=partial(custom_batch_size_scheduler,
                                 interval=5,
                                 increment=1),
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

# happy debugging!
trainer.train()
