from typing import Tuple, List, Dict

import torch
from transformers import AutoTokenizer
from openprompt.data_utils import InputExample
from torch.utils.data import Dataset


# MODEL_PATH = '../model/PLMs/mt0'
MODEL_PATH = '/Users/junon/model/PLMs/mt0'
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
# DEVICE = 'mps'

POLARITY_ID_TO_NAME = {
    -1: "negative",
    0: "neutral",
    1: "positive"
}

class PTDataset(Dataset):

    """
    PTDataset(object):
    |
    - pt_dataset(Dict):
      |
      - input_example(Openprompt_InputExample)
      - implicit_label(int)

    - polarities(List)
    """

    def __init__(self, path: str, implicit_symbol: bool):
        super(PTDataset, self).__init__()
        self.pt_datasets, self.polarities = self.read_corpus(
            path=path,
            implicit_symbol=implicit_symbol
        )

        # # load all aspects in the dataset
        # aspects = []
        # for input_example_dict in self.pt_datasets:
        #     aspects.append(input_example_dict['input_example'].text_b)
        #
        # self.aspect_list = aspects
        # self.aspect_set = list(set(aspects))

        # load the implicit labels (if given the "implicit symbol")
        if implicit_symbol:
            implicit_labels = []

            for current_input_example in self.pt_datasets:
                implicit_labels.append(current_input_example['implicit_label'])

            self.implicit_labels = implicit_labels

    def __getitem__(self, index):
        return self.pt_datasets[index], self.polarities[index]

    def __len__(self):
        return len(self.polarities)

    def read_corpus(self, path, implicit_symbol) -> Tuple[List[Dict], List[str]]:
        """
        Read the data of ABSA dataset and capture the List of InputExample of OpenPrompt package
        :param path: the path of dataset
        :param implicit_symbol: the symbol which denotes whether loading the dataset with implicit labels
        :return:
        """

        # init three lists for storing the input_examples (Openprompt.InputExample and implicit labels)
        # and the polarities(labels)
        datasets = []
        polarities = []

        with open(path, "r", encoding="utf-8") as file:
            lines = file.readlines()

        # handle the situation that loading the dataset with implicit labels
        if implicit_symbol:
            iteration = range(0, len(lines), 4)
            guid_divisor = 4
        else:
            iteration = range(0, len(lines), 3)
            guid_divisor = 3

        for i in iteration:
            # init a dictionary to store current data sample
            current_data = {}

            # get the context, which consists of two parts: the left context and right context of the aspect term
            text_left, _, text_right = [s.strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].strip()
            polarity = lines[i + 2].strip()

            # revert the whole sentence by text_left, text_right and aspect term
            full_context = text_left + ' ' + aspect + ' ' + text_right
            full_context = full_context.strip()

            if implicit_symbol:
                # store the current implicit label if passing the "implicit_symbol" arg
                implicit_label = lines[i + 3].strip()
                # the implicit label will be only 'Y' or 'N', so it's necessary to convert them to the int 1 or 0
                if implicit_label == 'Y':
                    current_data['implicit_label'] = 1
                elif implicit_label == 'N':
                    current_data['implicit_label'] = 0

            """
            !!! Importance
            """
            # for discriminative model: change the type of polarity to integer
            # for generative model: keep the text name of polarity
            # polarity = int(polarity) + 1
            polarity = POLARITY_ID_TO_NAME[int(polarity)]

            # construct the current input_example and store to the dict
            if i <= 0:
                current_input_example = InputExample(
                    guid=i,
                    text_a=full_context,
                    text_b=aspect,
                    label=polarity
                )
            else:
                current_input_example = InputExample(
                    guid=int(i / guid_divisor),
                    text_a=full_context,
                    text_b=aspect,
                    label=polarity
                )
            current_data['input_example'] = current_input_example

            # store to the corresponding list
            datasets.append(current_data)
            polarities.append(polarity)

        return datasets, polarities


def create_input_seq(current_input_example):

    current_input_seq = 'Given the sentence "' + current_input_example['input_example'].text_a + '", what is the sentiment polarity towards ' + current_input_example['input_example'].text_b + '?'
    return current_input_seq


def collect_fn(
        samples: Tuple[Dict, int],
        tokenizer,
        max_length: int
):

    # get the input_example and polarities
    # input_example_dicts： List[Dict[InputExample, int]]
    # polarities: List[str]
    input_example_dicts, polarities = zip(*samples)

    # init a list for storing implicit labels
    implicit_labels = []

    # create the input sequences
    input_seqs = list(
        map(lambda current_input_seq: create_input_seq(current_input_seq), input_example_dicts)
    )

    tokenized_input_seqs = tokenizer(
        input_seqs,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    tokenized_labels = tokenizer(
        list(polarities),
        max_length=3,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )['input_ids']

    # wrap and tokenize
    for item in input_example_dicts:

        # first, get the implicit labels from current samples (batched)
        implicit_labels.append(item['implicit_label'])

    tokenized_labels[tokenized_labels == tokenizer.pad_token_id] = -100

    # set the "label" attribute of tokenized_input_seqs
    # tokenized_example contains 3 keys:
    # "input_ids", "attention_mask", "labels"
    tokenized_input_seqs['labels'] = tokenized_labels

    return {
        "input_ids": tokenized_input_seqs['input_ids'],
        "attention_mask": tokenized_input_seqs['attention_mask'],
        "labels": tokenized_input_seqs['labels'],
        "implicit_labels": implicit_labels
    }


if __name__ == '__main__':

    # pass
    train_dataset = PTDataset(path='dataset/Laptops_Train_implicit_labeled.xml.seg', implicit_symbol=True)
    # print(list(train_dataset)[:5])
    # model, tokenizer, model_config, WrapClass = load_plm(
    #     model_name='t5',
    #     model_path=MODEL_PATH
    # )

    # wrapped_tokenizer = WrapClass(
    #     tokenizer=tokenizer,
    #     max_seq_length=128
    # )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    # wrapped_tokenizer =

    # template = ManualTemplate(
    #     tokenizer=tokenizer,
    #     text='{"placeholder":"text_a"} The {"placeholder":"text_b"} is: '
    # )

    batch = collect_fn(samples=list(train_dataset)[:10], tokenizer=tokenizer, max_length=128)
    # input()
    # print(train_dataset.pt_datasets[:10])

