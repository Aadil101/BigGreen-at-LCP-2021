from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import pickle5 as pkl
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('cmudict')
from nltk.stem import WordNetLemmatizer 
from nltk.tokenize import word_tokenize
from nltk import ngrams
lemmatizer = WordNetLemmatizer() 
from transformers import BertTokenizerFast
bert_tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)
import string
import ast
import json
from collections import Counter
import random
import re
import urllib
import requests
from functools import lru_cache
from itertools import product as iterprod
from itertools import chain, combinations
import cmudict
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

col_2_multi_dim = {'glove_word':300, 'glove_word_sum':300, 'glove_word_head':300, 'glove_word_tail':300, 'glove_context':300, 'infersent_embeddings':4096, 'elmo_word':1024, 'elmo_word_sum':1024, 'elmo_word_head':1024, 'elmo_word_tail':1024}
arpabet = nltk.corpus.cmudict.dict()

def ChunkIterator(documents):
    for document in documents:
        yield document

def get_tfidf_vectorizer(documents, lowercase=True, token_pattern=r'(?u)\b\w+\b', ngram_range=(1,1)):
    v = TfidfVectorizer(lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range)
    v.fit(ChunkIterator(documents))
    return v

def get_tf_vectorizer(documents, lowercase=True, token_pattern=r'(?u)\b\w+\b', ngram_range=(1,1), transform=False, tokenizer=None):
    v = CountVectorizer(lowercase=lowercase, token_pattern=token_pattern, ngram_range=ngram_range, tokenizer=tokenizer)
    if transform:
        x = v.fit_transform(ChunkIterator(documents))
        return v, x
    else:
        v.fit(ChunkIterator(documents))
        return v

def get_tfidf_score(x, v, r, default=1):
    idx = x['idx']
    token = x['token']
    if token in v.vocabulary_:
        return r[idx, v.vocabulary_[token]]
    else:
        if v.lowercase:
            token = token.lower()
            if token in v.vocabulary_:
                return r[idx, v.vocabulary_[token]]
            else:
                return default
        else:
            return default

def get_tf_score(x, v, r, lemmatize=False, default=0, tokenizer=None):
    token = x['token']
    if tokenizer:
        return sum([r[bpe.strip('#')] if bpe.strip('#') in r else 0 for bpe in tokenizer(token)])
    if lemmatize:
        token = lemmatizer.lemmatize(token)
    if token in r:
        return r[token]
    else:
        if v.lowercase:
            token = token.lower()
            if token in r:
                return r[token]
            else:
                return default
        else:
            return default

def tokenizer_wrapper(sentence):
    return [bpe.strip('#') for bpe in bert_tokenizer.tokenize(sentence)]

def get_num_OOV(x, v):
    sentence = x['sentence']
    return len([1 for word in word_tokenize(sentence) if word not in v.vocabulary_ and word not in string.punctuation])

def get_ngrams(sentence, token, N):
    return [' '.join(ngram) for ngram in ngrams(word_tokenize(sentence), N) if token in ngram]

def names(data):
    result = []
    for column in data:
        if column in ['id','corpus','sentence','token','complexity','class']:
            continue
        if column in col_2_multi_dim:
            for i in range(1, col_2_multi_dim[column]+1):
                result.append('{}_{}'.format(column, i))
        else:
            result.append(column)
    return result

def discrete(data):
    result = []
    for column in data:
        if column in ['id','corpus','sentence','token','complexity','class']:
            continue
        is_discrete = None
        if issubclass(data[column].dtype.type, np.integer):
            is_discrete = True
            #print(column, data[column].dtype, is_discrete)
        else:
            is_discrete = False
            #print(column, data[column].dtype, is_discrete)
        if column in col_2_multi_dim:
            for _ in range(col_2_multi_dim[column]):
                result.append(is_discrete)
        else:
            result.append(is_discrete)
    return result

def apply_pca(data, n_components=0.95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return scaler, pca, pca_data

def join_discovery_features(job, name, multi=False):    
    data = pd.read_csv('./data/'+job+'/macbook/'+name+'_p.tsv', sep='\t', index_col=0)
    for filename in tqdm(os.listdir('./data/'+job)):
        if not filename.startswith(name):
            continue
        if not filename.endswith('.tsv'):
            continue
        if filename == name+'_y.tsv':
            continue
        if filename == name+'_j.tsv':
            continue
        if multi:
            feature_name = filename.replace(name+'_','').replace('.tsv','').replace('_head','').replace('_tail','')
            feature = pd.read_csv('./data/'+job+'/'+filename,sep='\t',index_col=0)[feature_name]
            if feature_name+'_head' in filename:
                data[feature_name+'_head'] = feature
                if feature_name+'_tail' in data:
                    other_feature = pd.read_csv('./data/'+job+'/'+name+'_'+feature_name+'_tail.tsv',sep='\t',index_col=0)[feature_name]
                    data[feature_name+'_sum'] = feature + other_feature
            elif feature_name+'_tail' in filename:
                data[feature_name+'_tail'] = feature
                if feature_name+'_head' in data:
                    other_feature = pd.read_csv('./data/'+job+'/'+name+'_'+feature_name+'_head.tsv',sep='\t',index_col=0)[feature_name]
                    data[feature_name+'_sum'] = feature + other_feature
            else:
                if feature_name not in data:
                    data[feature_name] = feature
                else:
                    print('oops.')
        else:
            feature_name=filename.replace(name+'_','').replace('.tsv','')
            feature = pd.read_csv('./data/'+job+'/'+filename,sep='\t',index_col=0)[feature_name]
            #feature = feature.fillna(0)
            if feature_name not in data:
                data[feature_name] = feature
            else:
                print('oops.')
                #data[feature_name] = data.apply(lambda x: feature[x['id']] if np.isnan(x[feature_name]) and x['id'] in feature else x[feature_name], axis=1)
                #data[feature_name] = data[feature_name].fillna(0).astype(feature.dtype)
    data.to_csv('./data/'+job+'/'+name+'_y.tsv', sep='\t')

def log_transform(data, job, name, columns):
    if columns == 'all':
        for column in data:
            if column in ['id','corpus','sentence','token','complexity','class']:
                continue
            feature = data[column]
            if feature.dtype.type == np.object_:
                continue
            if not column.startswith('log_') and not column.startswith('Lg'):
                data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    elif columns in data:
        feature = data[columns]
        if not column.startswith('log_') and not column.startswith('Lg'):
            data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    elif isinstance(columns, list):
        for column in columns:
            feature = data[column]
            if not column.startswith('log_') and not column.startswith('Lg'):
                data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    data.to_csv('./data/'+job+'/'+name+'_y.tsv', sep='\t')

def join_macbook_discovery_features(job, name):
    discovery_data = pd.read_csv('./data/'+job+'/'+name+'_y.tsv', sep='\t', index_col=0)
    macbook_data = pd.read_csv('./data/'+job+'/macbook/'+name+'_o.tsv', sep='\t', index_col=0)
    if 'complexity' in macbook_data.columns:
        columns = ['id', 'corpus', 'sentence', 'token', 'complexity', 'class']
    else:
        columns = ['id', 'corpus', 'sentence', 'token']
    data = pd.merge(macbook_data, discovery_data, left_on=columns, right_on=columns)
    data.to_csv('./data/'+job+'/'+name+'_j.tsv', sep='\t')

def get_sorted_mi(path):
    mi = []
    with open(path, 'r') as file:
        for line in file:
            f, m = line.strip('\n').split(', ')
            m = float(m)
            mi.append((f, m))
    mi = sorted([(f, m) for (f, m) in mi], key=lambda x: x[1], reverse=True)
    return map(list, zip(*mi))

def custom_cv_folds(data, n_splits=5, test_size=0.1):
    std_idx = range(len(data))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_index, dev_index in sss.split(std_idx, data[['corpus', 'class']]):
        yield train_index, dev_index

def another_custom_cv_folds(data, n_splits=5, shuffle=True):
    std_idx = range(len(data))
    sss = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
    for train_index, dev_index in sss.split(std_idx, data['corpus']+'_'+data['class'].astype(str)):
        yield train_index, dev_index

def load_features_stacked(job, name, data=None, save=True):
    npy_path = './data/'+job+'/'+name+'_X.npy'
    features_stacked = None
    if os.path.isfile(npy_path):
        features_stacked = np.load(npy_path)
    else:
        path = './data/'+job+'/'+name+'_j.tsv'
        if data is None:
            data = pd.read_csv(path, sep='\t', index_col=0, converters={column: ast.literal_eval for column in col_2_multi_dim})
        else:
            for column in col_2_multi_dim:
                if column not in data:
                    continue
                data[column] = data[column].apply(ast.literal_eval)
        features = []
        for column in data:
            if column in ['id','corpus','sentence','token','complexity','class']:
                continue
            feature = data[column]
            if column in col_2_multi_dim:
                features.append(np.array(feature.tolist()))
            else:
                features.append(feature.to_numpy()[:, None])
        features_stacked = np.hstack(features)
        if save:
            np.save(npy_path, features_stacked)
    return features_stacked

def load_feature_name_2_idx(job, name, data=None):
    feature_name_2_idx_path = './data/'+job+'/'+name+'_name_2_idx.json'
    feature_name_2_idx = None
    if os.path.isfile(feature_name_2_idx_path):
        with open(feature_name_2_idx_path, 'r') as file:
            feature_name_2_idx = json.load(file)
    else:
        if data is None:
            path = './data/'+job+'/'+name+'_j.tsv'
            data = pd.read_csv(path, sep='\t', index_col=0, converters={column: ast.literal_eval for column in col_2_multi_dim})
        feature_names = names(data)
        feature_name_2_idx = dict(zip(feature_names, range(len(feature_names))))
        with open(feature_name_2_idx_path, 'w') as file:
            json.dump(feature_name_2_idx, file)
    return feature_name_2_idx

def MAE(y, y_pred):
    return mean_absolute_error(y, y_pred)

def pearson(y, y_pred):
    return pearsonr(y, y_pred)[0]

def pearson_p_val(y, y_pred):
    return pearsonr(y, y_pred)[1]

def spearman(y, y_pred):
    return spearmanr(y, y_pred)[0]

def preprocess_for_bert(data, output_path, do_round=True, do_mask=False, do_scale=False, do_token=True):
    if do_token:
        x = data[['complexity', 'sentence', 'token']]
    else:
        x = data[['complexity', 'sentence']]
    x.insert(0, 'idx', range(len(data)))
    if do_scale:
        x['complexity'] = data['complexity_scaled']
    if do_round:
        x['complexity'] = x['complexity'].map('{:.3f}'.format)
    if do_mask:
        x['sentence'] = x[['sentence', 'token']].apply(lambda y: '[MASK]'.join(y['sentence'].split(y['token'])), axis=1)
    x.to_csv(output_path, sep='\t', index=False, header=False, encoding='utf-8')

def class_2_index(samples, complexities, nclasses=5):
    result = samples.copy()
    for cls in samples.unique():
        samples_idx = samples == cls
        N = samples_idx.sum()
        bin_left = 1.0/nclasses*(cls-1)
        bin_right = bin_left+1.0/nclasses
        complexities_idx = complexities.between(bin_left, bin_right)
        mi, ma, bs, bks = buckets(complexities[complexities_idx], bin_left, bin_right) 
        buckelems = list(bks.elements())
        result[samples_idx] = makesample(N, buckelems, mi, ma, bs)
    return result

def buckets(discrete_set, amin=None, amax=None, bucket_size=None):
    if amin is None: amin=min(discrete_set)
    if amax is None: amax=min(discrete_set)
    if bucket_size is None: bucket_size = (amax-amin)/20
    def to_bucket(sample):
        if not (amin <= sample <= amax): return None  # no bucket fits
        return int((sample - amin) // bucket_size)
    b = Counter(to_bucket(s) for s in discrete_set if to_bucket(s) is not None)
    return amin, amax, bucket_size, b

def makesample(N, buckelems, mi, ma, bs):
    s = []
    for _ in range(N):
        buck = random.choice(buckelems)
        x = random.uniform(mi+buck*bs, mi+(buck+1)*bs)
        s.append(x)
    return s

def pearson_by_class(data, column):
    return data.groupby('class').apply(lambda x: pearson(x['complexity'], x[column]) if len(x) >= 2 else np.nan)

def pearson_by_corpus(data, column):
    return data.groupby('corpus').apply(lambda x: pearson(x['complexity'], x[column]) if len(x) >= 2 else np.nan)

def MAE_by_class(data, column):
    return data.groupby('class').apply(lambda x: MAE(x['complexity'], x[column]) if len(x) > 0 else np.nan)

def relevance(complexity, k=0.5):
    result = complexity.copy()
    q1 = np.percentile(complexity, 25)
    median = np.median(complexity)
    q3 = np.percentile(complexity, 75)

    r = q3-q1
    complexity_sorted = np.sort(complexity)
    adjL = complexity_sorted[complexity_sorted >= (q1-1.5*r)][0]
    adjH = complexity_sorted[complexity_sorted <= (q3+1.5*r)][-1]
    xa = adjL
    xc = adjH
    i = adjL*(adjL-complexity.min())/(median-complexity.min())
    d = adjH*(adjH-median)/(complexity.max()-median)
    m = (np.abs(median-(adjL+i)) + np.abs(median-(adjH-d)))/2
    xb = adjL+k*m
    xd = adjH-k*m
    c1 = np.exp(xa*np.log(1/3)/(xa-xb))
    s1 = np.log(1/3)/(xa-xb)
    c2 = np.exp(xc*np.log(1/3)/(xc-xd))
    s2 = np.log(1/3)/(xc-xd)

    less_idx = complexity <= median
    more_idx = complexity > median
    result[less_idx] = c1/(c1+np.exp(s1*complexity[less_idx]))
    result[more_idx] = c2/(c2+np.exp(s2*complexity[more_idx]))

    return result

def get_synth_cases(D, target, o=200, k=3, categorical_col = []):
    '''
    Function to generate the new cases.
    INPUT:
        D - pd.DataFrame with the initial data
        target - string name of the target column in the dataset
        o - oversampling rate
        k - number of nearest neighbors to use for the generation
        categorical_col - list of categorical column names
    OUTPUT:
        new_cases - pd.DataFrame containing new generated cases
    '''
    new_cases = pd.DataFrame(columns = D.columns) # initialize the list of new cases 
    ng = o // 100 # the number of new cases to generate
    for index, case in D.iterrows():
        # find k nearest neighbors of the case
        knn = KNeighborsRegressor(n_neighbors = k+1) # k+1 because the case is the nearest neighbor to itself
        knn.fit(D.drop(columns = [target]).values, D[[target]])
        neighbors = knn.kneighbors(case.drop(labels = [target]).values.reshape(1, -1), return_distance=False).reshape(-1)
        neighbors = np.delete(neighbors, np.where(neighbors == index))
        for i in range(0, ng):
            # randomly choose one of the neighbors
            x = D.iloc[neighbors[np.random.randint(k)]]
            attr = {}          
            for a in D.columns:
                # skip target column
                if a == target:
                    continue
                if a in categorical_col:
                    # if categorical then choose randomly one of values
                    if np.random.randint(2) == 0:
                        attr[a] = case[a]
                    else:
                        attr[a] = x[a]
                else:
                    # if continious column
                    diff = case[a] - x[a]
                    attr[a] = case[a] + np.random.randint(2) * diff
            # decide the target column
            new = np.array(list(attr.values()))
            d1 = cosine_similarity(new.reshape(1, -1), case.drop(labels = [target]).values.reshape(1, -1))[0][0]
            d2 = cosine_similarity(new.reshape(1, -1), x.drop(labels = [target]).values.reshape(1, -1))[0][0]
            attr[target] = (d2 * case[target] + d1 * x[target]) / (d1 + d2)
            
            # append the result
            new_cases = new_cases.append(attr,ignore_index = True)
                    
    return new_cases

def SmoteR(D, target, th = 0.999, o = 200, u = 100, k = 3, categorical_col = []):
    '''
    The implementation of SmoteR algorithm:
    https://core.ac.uk/download/pdf/29202178.pdf
    INPUT:
        D - pd.DataFrame - the initial dataset
        target - the name of the target column in the dataset
        th - relevance threshold
        o - oversampling rate
        u - undersampling rate
        k - the number of nearest neighbors
    OUTPUT:
        new_D - the resulting new dataset
    '''
    # median of the target variable
    y_bar = D[target].median()
    
    # find rare cases where target less than median
    rareL = D[(relevance(D[target]) > th) & (D[target] < y_bar)]  
    # generate rare cases for rareL
    new_casesL = get_synth_cases(rareL, target, o, k , categorical_col)
    
    # find rare cases where target greater than median
    rareH = D[(relevance(D[target]) > th) & (D[target] > y_bar)]
    # generate rare cases for rareH
    new_casesH = get_synth_cases(rareH, target, o, k , categorical_col)
    
    new_cases = pd.concat([new_casesL, new_casesH], axis=0)
    
    # undersample norm cases
    norm_cases = D[relevance(D[target]) <= th]
    # get the number of norm cases
    nr_norm = int(len(norm_cases) * u / 100)
    
    norm_cases = norm_cases.sample(min(len(D[relevance(D[target]) <= th]), nr_norm))
    
    # get the resulting dataset
    new_D = pd.concat([new_cases, norm_cases], axis=0)
    
    return new_D

def random_string(n=10):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))

def preprocess(data, path, scale=False):
    data.dropna(subset=['token'], inplace=True)
    if scale:
        data['complexity'] = minmax_scale(data['complexity'])
    if 'id' not in data.columns:
        data['id'] = [random_string(10) for _ in range(len(data))]
    if 'corpus' not in data.columns:
        data['corpus'] = np.nan
    data['class'] = pd.cut(data['complexity'], labels=[1,2,3,4,5], bins=[0,0.2,0.4,0.6,0.8,1], include_lowest=True)
    data.rename(columns={'subcorpus':'corpus'}, inplace=True)
    data.to_csv(path, sep='\t')

def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def multi_data(data):
    data_head = data.copy()
    data_head['token'] = data_head['token'].apply(lambda x: x.split(' ')[0])
    data_tail = data.copy()
    data_tail['token'] = data_tail['token'].apply(lambda x: x.split(' ')[1])
    return data_head, data_tail

def get_tf(word, exact=False):
    encoded_query = urllib.parse.quote(word) 
    params = {'corpus': 'eng-us', 'query': encoded_query, 'format': 'json'} 
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items()) 
    response = requests.get('https://api.phrasefinder.io/search?' + params) 
    response = json.loads(response.text)
    if exact:
        word_lst = word.split(' ')
        tks_lst = [(i, [tk['tt'] for tk in phrase['tks']]) for i, phrase in enumerate(response['phrases'])]
        for i, tks in tks_lst:
            if tks == word_lst:
                return response['phrases'][i]['mc']
        else:
            return 0
    else:
        return sum([phrase['mc'] for phrase in response['phrases']])

@lru_cache()
def wordbreak(s):
    s = s.lower()
    if s in arpabet:
        return arpabet[s]
    middle = len(s)/2
    partition = sorted(list(range(len(s))), key=lambda x: (x-middle)**2-x)
    for i in partition:
        pre, suf = (s[:i], s[i:])
        if pre in arpabet and wordbreak(suf) is not None:
            return [x+y for x,y in iterprod(arpabet[pre], wordbreak(suf))]
    return None

def get_phoneme_transition(mapping, stress_sensitive=True):
    if stress_sensitive:
        index = [['{}{}'.format(phoneme, i) for i in range(3)] if phoneme_type[0] == 'vowel' else [phoneme] for phoneme, phoneme_type in cmudict.phones()]
        index = [item for sublist in index for item in sublist] + ['UW']
        columns = index.copy()
    else:
        index, _ = map(list, zip(*cmudict.phones()))
        columns, _ = map(list, zip(*cmudict.phones()))
    table = pd.DataFrame(0, index=index, columns=columns)
    for input, output_map in mapping.items():
        for output, value in output_map.items():
            if stress_sensitive:
                table.at[input, output] += value
            else:
                table.at[input.strip('0').strip('1').strip('2'), \
                         output.strip('0').strip('1').strip('2')] += value
    table = table.div(table.sum(axis=1), axis=0).fillna(0)
    return table       

def get_char_transition(mapping, case_sensitive=True):
    if case_sensitive:
        index = list(string.ascii_lowercase+string.ascii_uppercase)
        columns = list(string.ascii_lowercase+string.ascii_uppercase)
    else:
        index = list(string.ascii_lowercase)
        columns = list(string.ascii_lowercase)
    table = pd.DataFrame(0, index=index, columns=columns)
    for input, output_map in mapping.items():
        if input.lower() not in index:
            continue
        for output, value in output_map.items():
            if output.lower() not in columns:
                continue
            if case_sensitive:
                table.at[input, output] += value
            else:
                table.at[input.lower(), output.lower()] += value
    table = table.div(table.sum(axis=1), axis=0).fillna(0)
    return table       

def get_single_char_transition(mapping):
    index = list(string.ascii_lowercase+string.ascii_uppercase)
    columns = list(string.ascii_lowercase+string.ascii_uppercase)+['[SEP]']
    table = pd.DataFrame(0, index=index, columns=columns)
    for input, output_map in mapping.items():
        if input not in index:
            continue
        for output, value in output_map.items():
            if output not in columns:
                continue
            table.at[input, output] += value
    table = table.div(table.sum(axis=1), axis=0).fillna(0)
    return table

def multi_data(data):
    data_head = data.copy()
    data_head['token'] = data_head['token'].apply(lambda x: x.split(' ')[0])
    data_tail = data.copy()
    data_tail['token'] = data_tail['token'].apply(lambda x: x.split(' ')[1])
    return data_head, data_tail       

def multi_compute(data, job, name, omit=set()):
    columns_with_lists = ['glove_word', 'glove_context', 'infersent_embeddings', 'elmo_word']
    path_head = './data/'+job+'/'+name+'_o_head.tsv'
    path_tail = './data/'+job+'/'+name+'_o_tail.tsv'
    data_head = pd.read_csv(path_head, sep='\t', index_col=0, converters={column: ast.literal_eval for column in columns_with_lists})
    data_tail = pd.read_csv(path_tail, sep='\t', index_col=0, converters={column: ast.literal_eval for column in columns_with_lists})
    for column in tqdm(data_head):
        if column in ['id','corpus','sentence','token','complexity','class']:
            continue
        feature_head, feature_tail = data_head[column], data_tail[column]
        if feature_head.dtype.type == np.object_:
            feature_head = np.array(feature_head.tolist())
            feature_tail = np.array(feature_tail.tolist())
        if column in omit:
            data[column] = feature_head.tolist()
        else:
            data[column+'_sum'] = (feature_head+feature_tail).tolist()
            data[column+'_head'] = feature_head.tolist()
            data[column+'_tail'] = feature_tail.tolist()
        
    data.to_csv('./data/'+job+'/'+name+'_o.tsv', sep='\t')

def save_other_features(data, config, job, name, multi=False):
    if multi:
        if 'complexity' in data:
            data_head, data_tail = multi_data(data[['id','corpus','sentence','token','complexity','class']])
        else:
            data_head, data_tail = multi_data(data[['id','corpus','sentence','token']])
        name_head = (name,'head')
        name_tail = (name,'tail')
        omit = save_other_features(data_head, config, job, name_head)
        _ = save_other_features(data_tail, config, job, name_tail)
        multi_compute(data, job, name, omit)
        return
    mode = type(name) is tuple
    omit = set()
    single_char_mapping = pkl.load(open(config['single_char_map_path'], 'rb'))
    single_char_transition = get_single_char_transition(single_char_mapping)
    char_mapping = pkl.load(open(config['char_transition_map_path'], 'rb'))
    char_transition = get_char_transition(char_mapping, case_sensitive=config['case_sensitive'])
    data['char_transition_lst'] = data['token'].apply(lambda x: [char_transition.loc[char_1 if config['case_sensitive'] else char_1.lower()][char_2 if config['case_sensitive'] else char_2.lower()] for char_1, char_2 in ngrams(x, 2)] if len(list(ngrams(x, 2))) != 0 else [single_char_transition.loc[x]['[SEP]']])
    data['char_transition_min'] = data['char_transition_lst'].apply(lambda x: min(x))
    data['char_transition_max'] = data['char_transition_lst'].apply(lambda x: max(x))
    data['char_transition_mean'] = data['char_transition_lst'].apply(lambda x: np.mean(x))
    data['char_transition_std'] = data['char_transition_lst'].apply(lambda x: np.std(x))
    del data['char_transition_lst']
    phoneme_mapping = pkl.load(open(config['phoneme_transition_map_path'], 'rb'))
    phoneme_transition = get_phoneme_transition(phoneme_mapping, stress_sensitive=config['stress_sensitive'])
    data['phoneme_transition_lst'] = data['token'].apply(lambda x: [[phoneme_transition.loc[phoneme_1 if config['stress_sensitive'] else phoneme_1.strip('0').strip('1').strip('2')][phoneme_2 if config['stress_sensitive'] else phoneme_2.strip('0').strip('1').strip('2')] for phoneme_1, phoneme_2 in ngrams(wb, 2)] if len(list(ngrams(wb, 2))) > 1 else [0] for wb in wordbreak(x)])
    data['phoneme_transition_min'] = data['phoneme_transition_lst'].apply(lambda x: min([probability for sublist in x for probability in sublist]))
    data['phoneme_transition_max'] = data['phoneme_transition_lst'].apply(lambda x: max([probability for sublist in x for probability in sublist]))
    data['phoneme_transition_mean'] = data['phoneme_transition_lst'].apply(lambda x: np.mean([probability for sublist in x for probability in sublist]))
    data['phoneme_transition_std'] = data['phoneme_transition_lst'].apply(lambda x: np.std([probability for sublist in x for probability in sublist]))
    del data['phoneme_transition_lst']
    if mode == True:
        google_ngram_1 = np.load('./data/'+job+'/'+name[0]+'_google_ngram_1_'+name[1]+'.npy')
    else:
        google_ngram_1 = np.load('./data/'+job+'/'+name+'_google_ngram_1.npy')
    data['google_ngram_1'] = google_ngram_1.ravel()
    if mode == True:
        google_ngram_2 = np.load('./data/'+job+'/'+name[0]+'_google_ngram_2_'+name[1]+'.npy')
    else:
        google_ngram_2 = np.load('./data/'+job+'/'+name+'_google_ngram_2.npy')
    data['google_ngram_2_head'] = google_ngram_2[:,0] 
    data['google_ngram_2_tail'] = google_ngram_2[:,1]
    data['google_ngram_2_min'] = data[['google_ngram_2_head', 'google_ngram_2_tail']].min(axis=1)
    data['google_ngram_2_max'] = data[['google_ngram_2_head', 'google_ngram_2_tail']].max(axis=1)
    data['google_ngram_2_mean'] = data[['google_ngram_2_head', 'google_ngram_2_tail']].mean(axis=1)
    data['google_ngram_2_std'] = data[['google_ngram_2_head', 'google_ngram_2_tail']].std(axis=1)
    if mode == True:
        google_ngram_3 = np.load('./data/'+job+'/'+name[0]+'_google_ngram_3_'+name[1]+'.npy')
    else:
        google_ngram_3 = np.load('./data/'+job+'/'+name+'_google_ngram_3.npy')
    data['google_ngram_3_head'] = google_ngram_3[:,0] 
    data['google_ngram_3_mid'] = google_ngram_3[:,1]
    data['google_ngram_3_tail'] = google_ngram_3[:,2]
    data['google_ngram_3_min'] = data[['google_ngram_3_head', 'google_ngram_3_mid', 'google_ngram_3_tail']].min(axis=1)
    data['google_ngram_3_max '] = data[['google_ngram_3_head', 'google_ngram_3_mid', 'google_ngram_3_tail']].max(axis=1)
    data['google_ngram_3_mean'] = data[['google_ngram_3_head', 'google_ngram_3_mid', 'google_ngram_3_tail']].mean(axis=1)
    data['google_ngram_3_std'] = data[['google_ngram_3_head', 'google_ngram_3_mid', 'google_ngram_3_tail']].std(axis=1)
    subtlex = pd.read_csv('./word_frequency/SUBTLEXus74286wordstextversion.tsv', sep='\t', index_col=0)
    df = data.copy()
    df['token_lower'] = df['token'].str.lower()
    df['token_upper'] = df['token'].str.capitalize()
    df = pd.merge(df, subtlex, how='left', left_on='token', right_index=True)
    df = pd.merge(df, subtlex, how='left', left_on='token_lower', right_index=True)
    for column in subtlex:
        df[column] = df[column+'_x'].where(df[column+'_x'].notnull(), df[column+'_y'])
        del df[column+'_x']
        del df[column+'_y']
    df = pd.merge(df, subtlex, how='left', left_on='token_upper', right_index=True)
    for column in subtlex:
        df[column] = df[column+'_x'].where(df[column+'_x'].notnull(), df[column+'_y'])
        del df[column+'_x']
        del df[column+'_y']
    del df['token_lower']
    del df['token_upper']
    for column in df:
        if column in data:
            continue
        data[column] = df[column]
    df = data.copy()
    frequencies, tokens = [], []
    with open('./word_frequency/all.num', 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                continue
            lst = line.strip().split()
            frequencies.append(int(lst[0]))
            tokens.append(lst[1])
    bnc = pd.DataFrame({'token':tokens, 'bnc_frequency':frequencies}).set_index('token')
    bnc = bnc.groupby('token').sum()
    df['token_lower'] = df['token'].str.lower()
    df['token_upper'] = df['token'].str.capitalize()
    df = pd.merge(df, bnc, how='left', left_on='token', right_index=True)
    df = pd.merge(df, bnc, how='left', left_on='token_lower', right_index=True)
    for column in bnc:
        df[column] = df[column+'_x'].where(df[column+'_x'].notnull(), df[column+'_y'])
        del df[column+'_x']
        del df[column+'_y']
    df = pd.merge(df, bnc, how='left', left_on='token_upper', right_index=True)
    for column in bnc:
        df[column] = df[column+'_x'].where(df[column+'_x'].notnull(), df[column+'_y'])
        del df[column+'_x']
        del df[column+'_y']
    del df['token_lower']
    del df['token_upper']
    for column in df:
        if column in data:
            continue
        data[column] = df[column]
    data.fillna(0, inplace=True)
    if mode == True:
        data.to_csv('./data/'+job+'/'+name[0]+'_o_'+name[1]+'.tsv', sep='\t')
    else:
        data.to_csv('./data/'+job+'/'+name+'_o.tsv', sep='\t')
    return omit

def get_predictions_with_reduced_data(df_train, X_train, y_train, X_test, frac_1=0.87, frac_2=0.95, frac_3=0.82, frac_4=0.0, frac_5=0.0, n=100):
    result = np.zeros((X_test.shape[0], n))
    df_train['iloc'] = range(len(df_train))
    for i in tqdm(range(n)):
        index = set(df_train.index)-set(df_train[df_train['class']==1].sample(frac=frac_1).index)-set(df_train[df_train['class']==2].sample(frac=frac_2).index)-set(df_train[df_train['class']==3].sample(frac=frac_3).index)-set(df_train[df_train['class']==4].sample(frac=frac_4).index)-set(df_train[df_train['class']==5].sample(frac=frac_5).index)
        df_train_reduced = df_train.loc[index]
        X_train_reduced = X_train[df_train_reduced['iloc'].to_numpy(), :]
        y_train_reduced = y_train[df_train_reduced['iloc'].to_numpy()]

        model = XGBRegressor(colsample_bytree=0.7,learning_rate=0.03,max_depth=5,min_child_weight=4,n_estimators=225,nthread=4,objective='reg:linear',silent=1,subsample=0.7) # 1e-3 and 0.3 is optimal
        #'colsample_bytree': 0.7, 'learning_rate': 0.03, 'max_depth': 5, 'min_child_weight': 4, 'n_estimators': 225, 'nthread': 4, 'objective': 'reg:linear', 'silent': 1, 'subsample': 0.7
        model = model.fit(X_train_reduced, y_train_reduced)
        y_test_pred = model.predict(X_test)
        result[:, i] = y_test_pred
    return result.mean(axis=1)

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))