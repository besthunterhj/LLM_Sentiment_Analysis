import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import create_input_seq
from openprompt.data_utils import InputExample


def inference(model, tokenizer, input_seq):

    model.eval()

    # tokenize
    tokenized_input_seq = tokenizer(input_seq, return_tensors='pt')
    # print(input_seq)
    # print(tokenized_input_seq)

    # inference
    with torch.no_grad():
        output = model.generate(input_ids=tokenized_input_seq['input_ids'], max_new_tokens=10)
        print(output)
        output_text = tokenizer.batch_decode(output.detach().cpu().numpy(), skip_special_tokens=True)
        print(output_text)


if __name__ == '__main__':

    # peft 的 LoRA 模块自动映射硬件位置？
    # ABSA inference
    # peft_model_path = 'mt0_LoRA_epoch_1_acc_0.4342'
    # config = PeftConfig.from_pretrained(peft_model_path)

    # model = AutoModelForSeq2SeqLM.from_pretrained('/Users/junon/model/PLMs/mt0')
    # model = PeftModel.from_pretrained(model, peft_model_path)
    # tokenizer = AutoTokenizer.from_pretrained('/Users/junon/model/PLMs/mt0')
    # current_input_example = InputExample(text_a='Can you buy any laptop that matches the quality of a MacBook ?', text_b='quality')
    # input_seq = create_input_seq(current_input_example={'input_example': current_input_example})
    # inference(model, tokenizer, input_seq=input_seq)

    # FSA inference
    peft_model_path = 'mt0_LoRA_epoch_1_acc_0.9471'
    config = PeftConfig.from_pretrained(peft_model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained('/Users/junon/model/PLMs/mt0')
    model = PeftModel.from_pretrained(model, peft_model_path)
    tokenizer = AutoTokenizer.from_pretrained('/Users/junon/model/PLMs/mt0')
    input_seq = 'According to Gran , the company has no plans to move all production to Russia , although that is where the company is growing .'
    inference(model, tokenizer, input_seq=input_seq)
