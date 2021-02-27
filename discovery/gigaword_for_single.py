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
name = 'single_test'
path = './data/'+job+'/macbook/'+name+'_p.tsv'
data = pd.read_csv(path, sep='\t', index_col=0)

print('loading paragraphs...')

external_paragraphs = []
char_transitions_map = {}
phoneme_transitions_map = {}
single_char_map = {}
punctuation = string.punctuation
digits = string.digits
letters = set(string.ascii_letters)
for subset in ['afe', 'apw', 'nyt', 'xie']:
#for subset in ['xie']:
    print(subset)
    folder_path = './gigaword_txt/{}/'.format(subset)
    for file_name in tqdm(os.listdir(folder_path)):
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            for line in file:
                external_paragraphs.append(line)
                '''
                line = line.strip().translate(str.maketrans('', '', punctuation)).translate(str.maketrans('', '', digits))
                for word in line.split():
                    if len(word) == 0:
                        continue
                    first_char = word[0]
                    if first_char in letters:
                        if first_char not in single_char_map:
                            single_char_map[first_char] = {}
                        if len(word) == 1:
                            if '[SEP]' not in single_char_map[first_char]:
                                single_char_map[first_char]['[SEP]'] = 0
                            single_char_map[first_char]['[SEP]'] += 1
                        else:
                            second_char = word[1]
                            if second_char not in single_char_map[first_char]:
                                single_char_map[first_char][second_char] = 0
                            single_char_map[first_char][second_char] += 1 
                    ###
                    for (char_1, char_2) in ngrams(word, 2):
                        if char_1 not in char_transitions_map:
                            char_transitions_map[char_1] = {}
                        if char_2 not in char_transitions_map[char_1]:
                            char_transitions_map[char_1][char_2] = 0
                        char_transitions_map[char_1][char_2] += 1
                    ###
                    wbs = wordbreak(word)
                    if wbs is None: continue
                    n = len(wbs)
                    if n == 0: continue
                    for wb in wbs:
                        for (phoneme_1, phoneme_2) in ngrams(wb, 2):
                            if phoneme_1 not in phoneme_transitions_map:
                                phoneme_transitions_map[phoneme_1] = {}
                            if phoneme_2 not in phoneme_transitions_map[phoneme_1]:
                                phoneme_transitions_map[phoneme_1][phoneme_2] = 0
                            phoneme_transitions_map[phoneme_1][phoneme_2] += 1.0/n
                '''

#pkl.dump(char_transitions_map, open('./data/'+job+'/char_transitions_map.pkl', 'wb'))
#pkl.dump(phoneme_transitions_map, open('./data/'+job+'/phoneme_transitions_map.pkl', 'wb'))
#pkl.dump(single_char_map, open('./data/'+job+'/single_char_map.pkl', 'wb'))

print('number of paragraphs:', len(external_paragraphs))

print('...done!')
print('computing tfidf or tf scores...')


#TF-IDF
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False)
result = v.transform(data['sentence'])
data['idx'] = range(len(data))
data['tfidf'] = data.apply(lambda x: get_tfidf_score(x, v, result), axis=1)
data.to_csv('./data/'+job+'/{}_tfidf.tsv'.format(name), sep='\t')


#TF
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
data['idx'] = range(len(data))
data['tf'] = data.apply(lambda x: get_tf_score(x, v, result), axis=1)
data.to_csv('./data/'+job+'/{}_tf.tsv'.format(name), sep='\t')


#TF of lemmatized aspect words
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
data['idx'] = range(len(data))
data['tf_lemma'] = data.apply(lambda x: get_tf_score(x, v, result, lemmatize=True), axis=1)
data.to_csv('./data/'+job+'/{}_tf_lemma.tsv'.format(name), sep='\t')


#Summed TF's of BPEs of token
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, tokenizer=tokenizer_wrapper)
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
data['idx'] = range(len(data))
data['tf_summed_bpe'] = data.apply(lambda x: get_tf_score(x, v, result, tokenizer=tokenizer_wrapper), axis=1)
data.to_csv('./data/'+job+'/{}_tf_summed_bpe.tsv'.format(name), sep='\t')


#count of OOV words
v = get_tf_vectorizer(external_paragraphs, lowercase=False)
data['idx'] = range(len(data))
data['num_OOV'] = data.apply(lambda x: get_num_OOV(x, v), axis=1)
data.to_csv('./data/'+job+'/{}_num_OOV.tsv'.format(name), sep='\t')

'''
#TF N-grams
N = 2
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, ngram_range=(N,N))
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
data['idx'] = range(len(data))
data['tf_ngram_{}'.format(N)] = data.apply(lambda x: np.sum([result[ngram] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in result], dtype=np.float64), axis=1).fillna(0)
data.to_csv('./data/'+job+'/{}_tf_ngram_{}.tsv'.format(name, N), sep='\t')


#TF N-grams
N = 3
v, x = get_tf_vectorizer(external_paragraphs, lowercase=False, transform=True, ngram_range=(N,N))
x_sum = x.sum(axis=0).tolist()[0]
result = {word: x_sum[idx] for word, idx in v.vocabulary_.items()}
data['idx'] = range(len(data))
data['tf_ngram_{}'.format(N)] = data.apply(lambda x: np.sum([result[ngram] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in result], dtype=np.float64), axis=1).fillna(0)
data.to_csv('./data/'+job+'/{}_tf_ngram_{}.tsv'.format(name, N), sep='\t')


#TF-IDF N-grams
N = 2
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False, ngram_range=(N,N))
result = v.transform(data['sentence'])
data['idx'] = range(len(data))
data['tfidf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[x['idx'], v.vocabulary_[ngram]] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in v.vocabulary_], dtype=np.float64), axis=1).fillna(0)
data.to_csv('./data/'+job+'/{}_tfidf_ngram_{}.tsv'.format(name, N), sep='\t')


#TF-IDF N-grams
N = 3
v = get_tfidf_vectorizer(external_paragraphs, lowercase=False, ngram_range=(N,N))
result = v.transform(data['sentence'])
data['idx'] = range(len(data))
data['tfidf_ngram_{}'.format(N)] = data.apply(lambda x: np.mean([result[x['idx'], v.vocabulary_[ngram]] for ngram in get_ngrams(x['sentence'], x['token'], N) if ngram in v.vocabulary_], dtype=np.float64), axis=1).fillna(0)
data.to_csv('./data/'+job+'/{}_tfidf_ngram_{}.tsv'.format(name, N), sep='\t')
'''
print('...boom!')