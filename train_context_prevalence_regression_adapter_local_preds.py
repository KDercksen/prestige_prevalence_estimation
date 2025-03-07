import json
import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch as t
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, set_peft_model_state_dict
from rich import print as rprint
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import check_random_state
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import wandb
from utils import read_json, read_jsonl


# compute sigmoid over numpy array
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# transformers compute_metrics function that calculates MAE and MSE
def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = sigmoid(preds.squeeze())
    labels = labels.squeeze()
    # data = wandb.Table(
    #     data=[[x, y] for (x, y) in zip(preds, labels)], columns=["preds", "labels"]
    # )
    # wandb.log({"scatter_plot": wandb.plot.scatter(data, "preds", "labels")})
    mae = np.mean(np.abs(preds - labels))
    mrae = np.mean(np.abs(preds - labels) / (labels + 1e-8))
    mse = np.mean((preds - labels) ** 2)
    return {"mae": mae, "mse": mse, "mrae": mrae}


# class that extends AdapterTrainer and implements the compute_loss function with t.nn.L1Loss
class L1LossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = t.sigmoid(outputs.logits)
        loss_fct = t.nn.L1Loss()
        loss = loss_fct(logits.view(-1), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# class that extends L1LossAdapterTrainer and implements hard negative mining by scaling the loss
class L1LossTrainerWithHardNegativeMining(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = t.sigmoid(outputs.logits)
        loss_fct = t.nn.L1Loss(reduction="none")
        loss = loss_fct(logits.view(-1), labels.view(-1))
        loss[labels > 0.1] *= 10
        loss = loss.mean()
        return (loss, outputs) if return_outputs else loss


# class that extends AdapterTrainer and implement the compute_loss function with a mean relative absolute error loss
class MRAELossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs["labels"]
        outputs = model(**inputs)
        logits = t.sigmoid(outputs.logits)
        loss = t.abs(logits - labels) / (labels + 1e-8)
        # normalize it to 0-1
        # loss = t.mean(loss / (1 + loss))
        loss = t.mean(loss)
        return (loss, outputs) if return_outputs else loss


class SavePEFTModelCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        pt_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pt_model_path):
            os.remove(pt_model_path)
        return control


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--prevalence_data_path", type=Path, required=True)
    p.add_argument(
        "--model_name",
        type=str,
        default="sentence-transformers/distiluse-base-multilingual-cased-v1",
    )
    p.add_argument("--batch_size", default=16, type=int)
    p.add_argument("--num_epochs", default=4, type=float)
    p.add_argument(
        "--model_save_path",
        type=Path,
        default=Path("models/training_sts_crossencoder_prevalence"),
    )
    p.add_argument("--filter_zero_labels", action="store_true")
    p.add_argument("--use_lora", action="store_true")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--use_global_labels", action="store_true")
    args = p.parse_args()

    check_random_state(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess(examples):
        if args.use_global_labels:
            inputs = tokenizer(
                examples["sentence"],
                truncation=True,
                max_length=min(tokenizer.model_max_length, 1024),
            )
        else:
            inputs = tokenizer(
                examples["sentence"],
                examples["context"],
                truncation=True,
                max_length=min(tokenizer.model_max_length, 1024),
            )
        return inputs

    data_args = read_json(args.prevalence_data_path / "args.json")
    thresholds = eval(data_args["thresholds"])

    rprint("Loading local prevalence scores...")
    y_local = t.load(args.prevalence_data_path / "local_prevalences_tensor.pkl")[
        :, 2
    ].float()
    if args.use_global_labels:
        y_local = t.load(args.prevalence_data_path / "global_prevalences_tensor.pkl")[
            :, 2
        ].float()

    print("original label size:", len(y_local))
    if args.filter_zero_labels:
        subselection = t.where(y_local > 0.0)[0]
    else:
        subselection = t.arange(len(y_local))
    y_local = y_local[subselection]
    print("label size after potential subselection:", len(y_local))
    # log transform
    y_local = t.log(y_local + 0.01)
    # scale the prevalences between 0 and 1
    y_local = (y_local - y_local.min()) / (y_local.max() - y_local.min())

    sample_idxs = read_json(args.prevalence_data_path / "sample_idxs.json")
    sample_idxs = [sample_idxs[idx] for idx in subselection]

    all_sentences = read_json("doc_bert_nli_full_index.ann-sent_map")
    all_reports = read_jsonl(
        "./data/Radboudumc_2000-2021_reports_anonymized_archive_v4_for_diag.json"
    )

    local_dataset = []
    report_idxs = set()
    for idx, yl in tqdm(zip(sample_idxs, y_local), total=len(sample_idxs)):
        sent = all_sentences[str(idx)]
        report = all_reports[sent["report_idx"]]
        report_idxs.add(sent["report_idx"])
        local_dataset.append(
            (
                {
                    "sentence": sent["text"],
                    "context": report["text"],
                    "label": yl.item(),
                },
                sent["report_idx"],
            )
        )

    report_idxs = sorted(list(report_idxs))
    print("Number of unique reports:", len(report_idxs))

    splitter = ShuffleSplit(n_splits=1, test_size=0.05)
    for i, (train_idx, test_idx) in enumerate(splitter.split(report_idxs)):
        print("running fold", i)
        # make this sets for speedy lookup later on
        train_reports = {report_idxs[x] for x in train_idx}
        dev_reports = {report_idxs[x] for x in test_idx[: len(test_idx) // 2]}
        test_reports = {report_idxs[x] for x in test_idx[len(test_idx) // 2 :]}

        train_samples = Dataset.from_list(
            [x[0] for x in local_dataset if x[1] in train_reports]
        ).map(preprocess, batched=True, remove_columns=["sentence", "context"])
        dev_samples = Dataset.from_list(
            [x[0] for x in local_dataset if x[1] in dev_reports]
        ).map(preprocess, batched=True, remove_columns=["sentence", "context"])
        test_samples = Dataset.from_list(
            [x[0] for x in local_dataset if x[1] in test_reports]
        ).map(preprocess, batched=True, remove_columns=["sentence", "context"])

        print("train samples:", len(train_samples))
        print("dev samples:", len(dev_samples))
        print("test samples:", len(test_samples))

        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=1
        )
        callbacks = []
        if args.use_lora:
            peft_config = LoraConfig.from_pretrained(
                args.model_save_path / "adapter_model"
            )
            # peft_config = LoraConfig(
            #     task_type=TaskType.SEQ_CLS,
            #     inference_mode=False,
            #     lora_alpha=32,
            #     lora_dropout=0.1,
            #     r=args.lora_r,
            #     target_modules=["query", "value"],
            # )
            model = get_peft_model(model, peft_config)
            state_dict = t.load(
                args.model_save_path / "adapter_model/adapter_model.bin"
            )
            set_peft_model_state_dict(model, state_dict)
            model.print_trainable_parameters()
            callbacks = [SavePEFTModelCallback()]
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_save_path, num_labels=1
            )

        training_args = TrainingArguments(
            overwrite_output_dir=True,
            output_dir=args.model_save_path / f"fold_{i}",
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=64 // args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=5e-5,
            warmup_ratio=0.1,
            evaluation_strategy="steps",
            eval_steps=200,
            save_steps=200,
            load_best_model_at_end=True,
            save_total_limit=1,
            metric_for_best_model="mae",
            greater_is_better=False,
            logging_steps=20,
            fp16=True,
        )
        trainer = L1LossTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_samples,
            eval_dataset=dev_samples,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        # trainer.train()
        predictions = trainer.predict(test_samples)
        metrics = predictions.metrics
        preds = sigmoid(predictions.predictions)
        label_ids = predictions.label_ids
        with open(args.model_save_path / "metrics.json", "w") as f:
            json.dump(metrics, f)
        np.save(
            args.model_save_path / "predictions.npy",
            preds,
        )
        np.save(args.model_save_path / "labels.npy", label_ids)
