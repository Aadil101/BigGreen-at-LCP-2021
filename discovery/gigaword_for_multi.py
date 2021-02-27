from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle5 as pkl
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk import ngrams
lemmatizer = WordNetLemmatizer() 
from transformers import BertTokenizerFast
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
import string
from utils import *

job = 'pickle'
name = 'multi_test'
path = './data/'+job+'/macbook/'+name+'_p.tsv'
#data = pd.read_csv(path, sep='\t', index_col=0)
df = pd.read_csv(path, sep='\t', index_col=0)
data_head, data_tail = multi_data(df)

print('loading paragraphs...')

external_paragraphs = []
for subset in ['afe', 'apw', 'nyt', 'xie']:
#for subset in ['xie']:
    print(subset)
    folder_path = './gigaword_txt/{}/'.format(subset)
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                external_paragraphs.append(line)
                    
print('number of paragraphs:', len(external_paragraphs))

print('...done!')
print('computing tfidf or tf scores...')


#TF-IDF
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False)
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    result = v.transform(data['sentence'])
    data['idx'] = range(len(data))
    data['tfidf'] = data.apply(lambda x: get_tfidf_score(x, v, result), axis=1)
    data.to_csv('./data/'+job+'/{}_tfidf_{}.tsv'.format(name, suffix), sep='\t')


#TF
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    data['idx'] = range(len(data))
    data['tf'] = data.apply(lambda x: get_tf_score(x, v, result), axis=1)
    data.to_csv('./data/'+job+'/{}_tf_{}.tsv'.format(name, suffix), sep='\t')


#TF of lemmatized aspect words
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    data['idx'] = range(len(data))
    data['tf_lemma'] = data.apply(lambda x: get_tf_score(x, v, result, lemmatize=True), axis=1)
    data.to_csv('./data/'+job+'/{}_tf_lemma_{}.tsv'.format(name, suffix), sep='\t')


#Summed TF's of BPEs of token
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, tokenizer=tokenizer_wrapper)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    data['idx'] = range(len(data))
    data['tf_summed_bpe'] = data.apply(lambda x: get_tf_score(x, v, result, tokenizer=tokenizer_wrapper), axis=1)
    data.to_csv('./data/'+job+'/{}_tf_summed_bpe_{}.tsv'.format(name, suffix), sep='\t')


#count of OOV words
v = get_tf_vectorizer(external_paragraphs, lowercase=False)
df['idx'] = range(len(df))
df['num_OOV'] = df.apply(lambda x: get_num_OOV(x, v), axis=1)
df.to_csv('./data/'+job+'/{}_num_OOV.tsv'.format(name, suffix), sep='\t')

'''
#TF N-grams
N = 2
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, ngram_range=(N,N))
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    data['idx'] = range(len(data))
    data['tf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[ngram] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in result], dtype=np.float64), axis=1).fillna(0)
    data.to_csv('./data/'+job+'/{}_tf_ngram_{}_{}.tsv'.format(name, N, suffix), sep='\t')


#TF N-grams
N = 3
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, ngram_range=(N,N))
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    data['idx'] = range(len(data))
    data['tf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[ngram] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in result], dtype=np.float64), axis=1).fillna(0)
    data.to_csv('./data/'+job+'/{}_tf_ngram_{}_{}.tsv'.format(name, N, suffix), sep='\t')


#TF-IDF N-grams
N = 2
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False, ngram_range=(N,N))
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    result = v.transform(data['sentence'])
    data['idx'] = range(len(data))
    data['tfidf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[x['idx'], v.vocabulary_[ngram]] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in v.vocabulary_], dtype=np.float64), axis=1).fillna(0)
    data.to_csv('./data/'+job+'/{}_tfidf_ngram_{}_{}.tsv'.format(name, N, suffix), sep='\t')


#TF-IDF N-grams
N = 3
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False, ngram_range=(N,N))
for data, suffix in [(data_head, 'head'), (data_tail, 'tail')]:
    result = v.transform(data['sentence'])
    data['idx'] = range(len(data))
    data['tfidf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[x['idx'], v.vocabulary_[ngram]] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in v.vocabulary_], dtype=np.float64), axis=1).fillna(0)
    data.to_csv('./data/'+job+'/{}_tfidf_ngram_{}_{}.tsv'.format(name, N, suffix), sep='\t')
'''

print('...boom!')