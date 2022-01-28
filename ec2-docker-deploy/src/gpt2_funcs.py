import streamlit as st
import torch
from torch import nn
from transformers import GPT2Model, GPT2Tokenizer

class SimpleGPT2SequenceClassifier(nn.Module):
    def __init__(self, hidden_size: int, num_classes:int ,max_seq_len:int, gpt_model_name:str):
        super(SimpleGPT2SequenceClassifier,self).__init__()
        self.gpt2model = GPT2Model.from_pretrained(gpt_model_name)
        self.fc1 = nn.Linear(hidden_size*max_seq_len, num_classes)
#         self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        """
        Args:
                input_id: encoded inputs ids of sent.
        """
        gpt_out, _ = self.gpt2model(input_ids=input_id, attention_mask=mask, return_dict=False)
        batch_size = gpt_out.shape[0]
        linear_output = self.fc1(gpt_out.view(batch_size,-1))
#         final_layer = self.relu(linear_output)   
        return linear_output

@st.cache
def load_model(path:str):
    model = SimpleGPT2SequenceClassifier(hidden_size=768, num_classes=5, max_seq_len=128, gpt_model_name="gpt2")
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model

def load_tokenizer():
    # load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def read_input(tokenizer, input_text):
    fixed_text = " ".join(input_text.lower().split())
    model_input = tokenizer(fixed_text, padding='max_length', max_length=128, truncation=True, return_tensors="pt")
    mask = model_input['attention_mask'].cpu()
    input_id = model_input["input_ids"].squeeze(1).cpu()
    return input_id, mask

def run_model(model, input_id, mask):
    classes = ["business", "entertainment", "sport", "tech", "politics"]
    output = model(input_id, mask)
    prob = torch.nn.functional.softmax(output, dim=1)[0]
    _, indices = torch.sort(output, descending=True)
    return {classes[idx]: prob[idx].item() for idx in indices[0][:5]}