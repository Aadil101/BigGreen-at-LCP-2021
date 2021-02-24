from transformers import *
from tqdm import tqdm
import torch

device = 'cuda'
model_id = 'gpt2-large'
config = GPT2Config.from_pretrained(model_id)
model = GPT2LMHeadModel.from_pretrained(model_id, config=config).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
# model = BertLMHeadModel.from_pretrained(model_id).to(device)
# tokenizer = BertTokenizerFast.from_pretrained(model_id)

# load data
import pandas as pd
files = ['bea19-test.tsv']
encodings = []
for f in files:
    if f.endswith('.tsv'):
        data = pd.read_csv(f, sep='\t', header=0)
    else:
        data = pd.read_csv(f, header=0)
    # text = [data['text'].tolist()[i] for i in range(len(data)) if data['grammatical'].tolist()[i] == 1]
    text = data['text'].tolist()
    encodings += [tokenizer(var, return_tensors='pt') for var in text]

lls = []
for i in tqdm(range(len(encodings))):
    input_ids = encodings[i].input_ids.to(device)
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        try:
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0]
            lls.append(log_likelihood)
        except:
            print(input_ids)
            lls.append(-1)

    # lls.append(log_likelihood)
# print(lls)
# ppl = torch.exp(torch.stack(lls).sum() / i)
# print(ppl)
ppls = [torch.exp(var).cpu().data.tolist() if var != -1 else -1 for var in lls]
df = pd.DataFrame(ppls, columns=['ppl'])
df.to_csv('ppl.csv', index=False)
