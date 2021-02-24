import json
import pandas as pd
from transformers import *

source = 'bea19'
out = 'bea19'
model = 'bert-large-cased'

tokenizer = BertTokenizer.from_pretrained(model)
train = pd.read_csv('{}-train.tsv'.format(source), sep='\t', header=0)
dev = pd.read_csv('{}-dev.tsv'.format(source), sep='\t', header=0)
test = pd.read_csv('{}-test.tsv'.format(source), sep='\t', header=0)

text = train['text'].tolist()
labels = train['label'].tolist()
with open('{}_train.json'.format(out), 'a') as f:
	for i in range(len(text)):
		line_dict = {}
		line_dict["uid"] = str(i)
		line_dict["label"] = labels[i]
		if type(text[i]) == str:
			tokenization_res = tokenizer.encode_plus(text[i])
			line_dict["token_id"] = tokenization_res['input_ids']
			line_dict["type_id"] = tokenization_res['token_type_ids']
			json.dump(line_dict, f)
			f.write('\n')

text = dev['text'].tolist()
labels = dev['label'].tolist()
with open('{}_dev.json'.format(out), 'a') as f:
	for i in range(len(text)):
		line_dict = {}
		line_dict["uid"] = str(i)
		line_dict["label"] = labels[i]
		if type(text[i]) == str:
			tokenization_res = tokenizer.encode_plus(text[i])
			line_dict["token_id"] = tokenization_res['input_ids']
			line_dict["type_id"] = tokenization_res['token_type_ids']
			json.dump(line_dict, f)
			f.write('\n')

text = test['text'].tolist()
labels = test['label'].tolist()
with open('{}_test.json'.format(out), 'a') as f:
	for i in range(len(text)):
		line_dict = {}
		line_dict["uid"] = str(i)
		line_dict["label"] = labels[i]
		if type(text[i]) == str:
			tokenization_res = tokenizer.encode_plus(text[i])
			line_dict["token_id"] = tokenization_res['input_ids']
			line_dict["type_id"] = tokenization_res['token_type_ids']
			json.dump(line_dict, f)
			f.write('\n')