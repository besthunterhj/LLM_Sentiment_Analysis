import torch
from peft import LoraConfig, get_peft_model, LoraConfig, TaskType
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from utils import PTDataset, collect_fn, DEVICE


# MODEL_PATH = '/Users/junon/model/PLMs/mt0'
MODEL_PATH = '/Users/junon/model/PLMs/flan_t5_large'
# MODEL_PATH = '../model/PLMs/mt0'
LABEL_MAPPING = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}


def evaluate(model, test_dataloader, tokenizer, test_polarities):

    # avoid the effect of the built-in dropout of the model
    model.eval()

    eval_preds = []

    # n_correct and n_total: total amount of the correct results and all results
    n_correct, n_total = 0, 0

    # n_implicit_correct and n_implicit_total: for the implicit test examples, the amount of correct and all results
    n_implicit_correct, n_implicit_total = 0, 0

    # targets_all and outputs_all are lists, they store all true labels and prediction labels correspondingly
    targets_all, outputs_all = [], []

    # targets_implicit_all and outputs_implicit_all are lists for the implicit examples
    # they store all true labels and prediction labels correspondingly
    targets_implicit_all, outputs_implicit_all = [], []

    with torch.no_grad():
        for i_batch, batch in tqdm(enumerate(test_dataloader)):

            # get the input
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            implicit_labels = batch['implicit_labels']

            eval_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            # decode the prediction
            eval_preds.extend(
                tokenizer.batch_decode(
                    torch.argmax(eval_outputs.logits, -1).detach().cpu().numpy(),
                    skip_special_tokens=True
                )
            )

            start = i_batch * labels.shape[0]
            n_total += labels.shape[0]
            for i in range(labels.shape[0]):
                true_labels = test_polarities[start:(start+labels.shape[0])]
                targets_all.append(LABEL_MAPPING[true_labels[i].strip()])
                if eval_preds[i].strip() not in ['positive', 'neutral', 'negative']:
                    outputs_all.append(LABEL_MAPPING['neutral'])
                else:
                    outputs_all.append(LABEL_MAPPING[eval_preds[i].strip()])

                if eval_preds[i].strip() == true_labels[i].strip():
                    n_correct += 1

    acc = n_correct / n_total
    try:
        f1 = f1_score(
            y_true=targets_all,
            y_pred=outputs_all,
            labels=[0, 1, 2],
            average='macro'
        )
    except ValueError:
        print("targets_all: ", targets_all)
        print("outputs_all: ", outputs_all)
        return acc, 0

    return acc, f1


def train(model, num_epochs, train_dataloader, test_dataloader, optimizer, scheduler, tokenizer, test_polarities):
    # for each epoch, validate the model after the training
    best_acc = 0

    for epoch in range(num_epochs):
        # training step
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader):
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

        val_acc, val_f1 = evaluate(
            model=model,
            test_dataloader=test_dataloader,
            tokenizer=tokenizer,
            test_polarities=test_polarities
        )
        print('> val_acc: {:.4f}, val_f1: {:.4f}'.format(val_acc, val_f1))

        if val_acc > best_acc:
            best_acc = val_acc
            peft_model_id = f"mt0_LoRA_epoch_{epoch}_acc_{round(val_acc, 4)}"
            model.save_pretrained(peft_model_id)
            print(f'{peft_model_id} saving Done!')


def main(batch_size, max_len, lr, num_epochs):

    train_dataset = PTDataset(path='dataset/Laptops_Train_implicit_labeled.xml.seg', implicit_symbol=True)
    test_dataset = PTDataset(path='dataset/Laptops_Test_Gold_implicit_labeled.xml.seg', implicit_symbol=True)

    # init the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    # init the collection function
    collect_fc = lambda samples: collect_fn(
        samples=samples,
        tokenizer=tokenizer,
        max_length=max_len
    )

    # init the dataloader
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collect_fc
    )

    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collect_fc
    )

    # set the peft config (here, we choose LoRA)
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
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

    evaluate(model=model, test_dataloader=test_dataloader, tokenizer=tokenizer, test_polarities=test_dataset.polarities)

    # train and validate
    # train(
    #     model=model,
    #     num_epochs=num_epochs,
    #     train_dataloader=train_dataloader,
    #     test_dataloader=test_dataloader,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     tokenizer=tokenizer,
    #     test_polarities=test_dataset.polarities
    # )


if __name__ == '__main__':

    main(
        batch_size=16,
        max_len=128,
        lr=1e-5,
        num_epochs=3
    )
