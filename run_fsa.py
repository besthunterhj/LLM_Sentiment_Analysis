import numpy as np
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup, default_data_collator
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import os

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_PATH = '../model/PLMs/mt0'


def get_dataset(dataset_name: str):

    # load the dataset
    dataset = load_dataset(dataset_name, "sentences_allagree")
    # the original dataset only contains the train set
    dataset = dataset["train"].train_test_split(test_size=0.1)
    dataset["validation"] = dataset["test"]
    del dataset["test"]

    # classes: ['negative', 'neutral', 'positive']
    classes = dataset["train"].features["label"].names
    dataset = dataset.map(
        lambda x: {"text_label": [classes[label] for label in x["label"]]},
        batched=True,
        num_proc=1
    )

    return dataset


def preprocess_function(examples, tokenizer, max_length=128):

    input_seqs = examples['sentence']
    # labels are string not integers, because the output of the model should be the name of label not id
    labels = examples['text_label']

    # tokenize
    tokenized_input_seqs = tokenizer(
        input_seqs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # the tokenized labels should only contain the input_ids
    tokenized_labels = tokenizer(
        labels,
        max_length=3,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )['input_ids']

    tokenized_labels[tokenized_labels == tokenizer.pad_token_id] = -100

    # set the "label" attribute of tokenized_input_seqs
    tokenized_input_seqs['labels'] = tokenized_labels

    return tokenized_input_seqs


def train_eval(model, num_epochs, train_loader, test_loader, optimizer, scheduler):
    # for each epoch, validate the model after the training
    best_acc = 0

    for epoch in range(num_epochs):
        # training step
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader):
            # clear the gradient accumulators
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # get the loss
            current_batch_loss = outputs.loss
            total_loss += current_batch_loss.detach().float()

            # update
            current_batch_loss.backward()
            optimizer.step()
            scheduler.step()

        # validation step
        model.eval()
        eval_loss = 0
        eval_preds = []
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            with torch.no_grad():
                eval_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

            # get the eval loss
            loss = eval_outputs.loss
            eval_loss += loss.detach().float()

            # decode the prediction
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(eval_outputs.logits, -1).detach().cpu().numpy(),
                    skip_special_tokens=True
                )
            )

        eval_epoch_loss = eval_loss / len(test_loader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_loader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")

        # measure the metrics for each epoch
        correct = 0
        total = 0
        for pred, true in zip(eval_preds, dataset["validation"]["text_label"]):
            if pred.strip() == true.strip():
                correct += 1
            total += 1
        accuracy = correct / total
        print(f"Accuracy: {accuracy}")
        print(f"{eval_preds[:10]=}")
        print(f"{dataset['validation']['text_label'][:10]=}")

        if accuracy > best_acc:
            best_acc = accuracy
            peft_model_id = f"mt0_LoRA_epoch_{epoch}_acc_{round(accuracy, 4)}"
            model.save_pretrained(peft_model_id)
            print(f'{peft_model_id} saving Done!')


def main(dataset, tokenizer, batch_size, lr, num_epochs):

    # set the peft config (here, we choose LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )

    preprocess_fc = lambda x: preprocess_function(
        examples=x,
        tokenizer=tokenizer
    )

    # tokenize and convert to the tensor
    processed_dataset = dataset.map(
        preprocess_fc,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_dataset['train']
    validate_dataset = processed_dataset['validation']

    # init the data loader
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=default_data_collator,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        validate_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
        pin_memory=True
    )

    # init the PLM
    # the trainable parameters are set as query and values of attention blocks
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path=MODEL_PATH)
    model = get_peft_model(model=model, peft_config=peft_config)
    model.print_trainable_parameters()
    model = model.to(DEVICE)

    # set the optimizer
    optimizer = AdamW(params=model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0.1 * len(train_dataloader) * num_epochs,
        num_training_steps=len(train_dataloader) * num_epochs
    )

    train_eval(
        model=model,
        num_epochs=num_epochs,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        optimizer=optimizer,
        scheduler=scheduler
    )


if __name__ == '__main__':

    dataset = get_dataset(dataset_name="financial_phrasebank")
    # print(dataset['train'])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # preprocess_function(examples=dataset['train'][:5], tokenizer=tokenizer)

    # A crucial problem is the generated texts are not constrained into 3 classes:['negative', 'neutral', 'positive']
    main(dataset=dataset, tokenizer=tokenizer, batch_size=16, lr=1e-3, num_epochs=3)
