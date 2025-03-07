"""
The system trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) on the SNLI + MultiNLI (AllNLI) dataset
with MultipleNegativesRankingLoss. Entailnments are poisitive pairs and the contradiction on AllNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset

Usage:
python training_nli_v2.py

OR
python training_nli_v2.py pretrained_transformer_model_name
"""
import logging
import math
import random
import sys
from datetime import datetime

from sentence_transformers import (
    InputExample,
    LoggingHandler,
    SentenceTransformer,
    datasets,
    losses,
    models,
)
from sentence_transformers.evaluation import TripletEvaluator

#### Just some code to print debug information to stdout
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    handlers=[LoggingHandler()],
)
#### /print debug information to stdout

model_name = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "models/bert-base-multilingual-cased-finetuned-full"
)
train_batch_size = 1  # The larger you select this, the better the results (usually). But it requires more GPU memory
max_seq_length = 75
num_epochs = 4

# Save path of the model
model_save_path = (
    "models/training_nli_v2_"
    + model_name.replace("/", "-")
    + "-"
    + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)


# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(
    word_embedding_model.get_word_embedding_dimension(), pooling_mode="mean"
)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Check if dataset exsist. If not, download and extract  it
nli_dataset_path = "data/mednli/"

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read MedNLI train dataset")


def add_to_samples(data, sent1, sent2, label):
    if sent1 not in data:
        data[sent1] = {
            "contradiction": set(),
            "entailment": set(),
            "neutral": set(),
        }
    data[sent1][label].add(sent2)


train_data = {}
with open(nli_dataset_path + "MEDNLI_train_dutch_dl.txt", encoding="utf8") as f:
    for line in f:
        sents = line.split("\t")
        sent1 = sents[0].strip()
        sent2 = sents[1].strip()
        label = sents[2].strip()

        add_to_samples(train_data, sent1, sent2, label)
        add_to_samples(train_data, sent2, sent1, label)  # Also add the opposite


train_samples = []
for sent1, others in train_data.items():
    if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
        train_samples.append(
            InputExample(
                texts=[
                    sent1,
                    random.choice(list(others["entailment"])),
                    random.choice(list(others["contradiction"])),
                ]
            )
        )
        train_samples.append(
            InputExample(
                texts=[
                    random.choice(list(others["entailment"])),
                    sent1,
                    random.choice(list(others["contradiction"])),
                ]
            )
        )

logging.info("Train samples: {}".format(len(train_samples)))


# Special data loader that avoid duplicates within a batch
train_dataloader = datasets.NoDuplicatesDataLoader(
    train_samples, batch_size=train_batch_size
)

# Our training loss
train_loss = losses.MultipleNegativesRankingLoss(model)

logging.info("Read MEDNLI dev dataset")
dev_data = {}
with open(nli_dataset_path + "MEDNLI_dev_dutch_dl.txt", encoding="utf8") as f:
    for line in f:
        sents = line.split("\t")
        sent1 = sents[0].strip()
        sent2 = sents[1].strip()
        label = sents[2].strip()
        add_to_samples(dev_data, sent1, sent2, label)
        add_to_samples(dev_data, sent2, sent1, label)

dev_samples = []
for sent1, others in dev_data.items():
    if len(others["entailment"]) > 0 and len(others["contradiction"]) > 0:
        dev_samples.append(
            [
                sent1,
                random.choice(list(others["entailment"])),
                random.choice(list(others["contradiction"])),
            ]
        )
        dev_samples.append(
            [
                random.choice(list(others["entailment"])),
                sent1,
                random.choice(list(others["contradiction"])),
            ]
        )

logging.info("Dev samples: {}".format(len(dev_samples)))

dev_evaluator = TripletEvaluator(
    [x[0] for x in dev_samples],
    [x[1] for x in dev_samples],
    [x[2] for x in dev_samples],
    name="triplet-eval",
)


# Configure the training
warmup_steps = math.ceil(
    len(train_dataloader) * num_epochs * 0.2
)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    evaluator=dev_evaluator,
    epochs=num_epochs,
    evaluation_steps=int(len(train_dataloader) * 0.1),
    warmup_steps=warmup_steps,
    output_path=model_save_path,
    use_amp=True,  # Set to True, if your GPU supports FP16 operations
    optimizer_params={"lr": 1e-5},
)

# Test examples
test_samples = []
with open(nli_dataset_path + "MEDNLI_test_dutch_dl.txt", encoding="utf8") as f:
    for line in f:
        sents = line.split("\t")
        sent1 = sents[0].strip()
        sent2 = sents[1].strip()
        label = sents[2].strip()
        test_samples.append([sent1, sent2, label])

model = SentenceTransformer(model_save_path)
test_evaluator = TripletEvaluator(
    [x[0] for x in test_samples],
    [x[1] for x in test_samples],
    [x[2] for x in test_samples],
    name="triplet-eval-test",
)
test_evaluator(model, output_path=model_save_path)
