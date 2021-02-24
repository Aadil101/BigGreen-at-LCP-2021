import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk import ngrams
from sklearn.preprocessing import StandardScaler, minmax_scale
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import torch
import pickle as pkl
from lib.InferSent.models import InferSent
import pandas as pd
import syllables
import inspect
import textstat
import readability
from allennlp.modules.elmo import Elmo, batch_to_ids
from nltk.data import load
from string import punctuation
import os
import ast
from scipy.stats import pearsonr, spearmanr
import math
import urllib
import requests
import json


def glove_word(data, embeddings_dict, lower=True):
    embeddings = np.zeros((len(data), 300))
    for i, item in enumerate(data['token']):
        if lower:
            token = str(item).lower()
        else:
            token = str(item)
        found = 0
        for sub_token in token.split():
            if sub_token in embeddings_dict:
                found += 1
                embeddings[i] += embeddings_dict[sub_token]
            else:
                embeddings[i] += np.zeros(300)
        if found > 0:
            embeddings[i] /= found
    #print('found {}, missing {}'.format(y, n))
    return embeddings

def glove_context(data, embeddings_dict, lower=True):
    embeddings = np.zeros((len(data), 300))
    for i, sentence in enumerate(data['sentence']):
        found = 0
        for item in word_tokenize(sentence):
            if lower:
                token = str(item).lower()
            else:
                token = str(item)
            if token in embeddings_dict:
                found += 1
                embeddings[i] += embeddings_dict[token]
        if found > 0:
            embeddings[i] /= found
    return embeddings

def glove(data, embeddings_dict, lower=True):
    return np.concatenate((glove_word(data, embeddings_dict, lower), \
                           glove_context(data, embeddings_dict, lower)), \
                           axis=1)

def get_POS(sentence, token, debug=False):
    word_tokens = word_tokenize(sentence)
    pos_tags = pos_tag(word_tokens)
    if token in word_tokens:
        idx = word_tokens.index(token)
    else:
        if debug:
            return np.nan
        for word_token in word_tokens:
            if token in word_token:
                break
        else:
            return np.nan
        return get_POS(sentence.replace(word_token, token), token, debug=True)
    return pos_tags[idx][1]

def get_hyponyms(sentence, word, disambiguate=True):
    if disambiguate:
        l = lesk(sentence, word)
        if not l:
            return []
        return [hyponym.name() for hyponym in l.hyponyms()]
    else:
        hyponyms = []
        for ss in wn.synsets(word):
            for hyponym in ss.hyponyms():
                hyponyms.append(hyponym.name())
        return hyponyms
        
def get_hypernyms(sentence, word, disambiguate=True):
    if disambiguate:
        l = lesk(sentence, word)
        if not l:
            return []
        return [hypernym.name() for hypernym in l.hypernyms()]
    else:
        hypernyms = []
        for ss in wn.synsets(word):
            for hypernym in ss.hypernyms():
                hypernyms.append(hypernym.name())
        return hypernyms

def token_depth(parse, token):
    n_leaves = len(parse.leaves())
    leavepos = set(parse.leaf_treeposition(n) for n in range(n_leaves))
    for pos in parse.treepositions():
        if pos in leavepos and parse[pos] == token:
            return len(pos)
    else:
        return parse.height()

def num_words_at_depth(parse, depth):
    n_leaves = len(parse.leaves())
    leavepos = set(parse.leaf_treeposition(n) for n in range(n_leaves))
    result = 0
    for pos in parse.treepositions():
        if pos in leavepos and len(pos) == depth:
            result += 1
    return result

def apply_pca(data, n_components=0.95):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return scaler, pca, pca_data

def preprocess(data, path, scale=False):
    data.dropna(subset=['token'], inplace=True)
    if scale:
        data['complexity'] = minmax_scale(data['complexity'])
    if 'complexity' in data.columns:
        data['class'] = pd.cut(data['complexity'], labels=[1,2,3,4,5], bins=[0,0.2,0.4,0.6,0.8,1], include_lowest=True)
    data.rename(columns={'subcorpus':'corpus'}, inplace=True)
    data.to_csv(path, sep='\t')

def build_glove_embeddings(config):
    result = {}
    with open(config['glove_path'], 'r', encoding="utf-8") as f:
        for line in tqdm(f):
            values = line.split()
            word = ''.join(values[:-300])
            coefs = np.asarray(values[-300:], dtype='float32')
            result[word] = coefs
    return result

def build_infersent_embeddings(data, config):
    V = config['infersent_V']
    MODEL_PATH = config['infersent_MODEL_PATH'] % V
    W2V_PATH = config['infersent_W2V_PATH']
    params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
    infersent = InferSent(params_model)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    infersent.set_w2v_path(W2V_PATH)
    if config['infersent_lower']:
        sentences = data['sentence'].str.lower().tolist()
    else:
        sentences = data['sentence'].tolist()
    infersent.build_vocab(sentences, tokenize=True)
    result = infersent.encode(sentences, tokenize=True)
    return result

def batches(iterable, n=1):
    l = len(iterable)
    result = []
    for ndx in range(0, l, n):
        result.append(iterable[ndx:min(ndx + n, l)])
    return result

def save_embeddings(data, glove_embeddings, config, path, context=True, multi=False):
    if multi:
        data_head, data_tail = multi_data(data)
        base, ext = os.path.splitext(path)
        path_head = os.path.join(base+'_head'+ext)
        path_tail = os.path.join(base+'_tail'+ext)
        omit = save_embeddings(data_head, glove_embeddings, config, path_head, context=context)
        _    = save_embeddings(data_tail, glove_embeddings, config, path_tail, context=context)
        multi_compute(data, path, path_head, path_tail, omit)
        return
    data['glove_word'] = glove_word(data, glove_embeddings, lower=config['glove_lower']).tolist()
    omit = set()
    if context:
        data['glove_context'] = glove_context(data, glove_embeddings, lower=config['glove_lower']).tolist()
        data['infersent_embeddings']=build_infersent_embeddings(data, config).astype(np.float16).tolist()
        omit.add('glove_context'); omit.add('infersent_embeddings')
        elmo = Elmo(config['options_file'], config['weight_file'], 1, dropout=0)
        embeddings_lst = []
        sentences = [word_tokenize(sentence) for sentence in data['sentence']]
        for batch in tqdm(batches(sentences, 100)):
            character_ids = batch_to_ids(batch)
            embeddings = elmo(character_ids)
            embeddings_lst.append(embeddings)
        elmo = np.zeros((len(sentences), 1024))
        idx, count = 0, 0
        for embeddings in embeddings_lst:
            e = embeddings['elmo_representations'][0].detach().numpy()
            for i in range(e.shape[0]):
                token = data.iloc[idx]['token']
                for j, word in enumerate(sentences[idx]):
                    if token == word or token in word:
                        break
                else:
                    j = None
                    count += 1
                if j:
                    elmo[idx] = e[i, j]
                else:
                    elmo[idx] = 0
                idx += 1
        data['elmo_word'] = elmo.tolist()
    data.to_csv(path, sep='\t')
    return omit

def save_other_features(data, parse_lst_path, config, path, context=True, parse=True, multi=False):
    if multi:
        if 'complexity' in data:
            data_head, data_tail = multi_data(data[['id','corpus','sentence','token','complexity','class']])
        else:
            data_head, data_tail = multi_data(data[['id','corpus','sentence','token']])
        base, ext = os.path.splitext(path)
        path_head = base+'_head'+ext
        path_tail = base+'_tail'+ext
        omit = save_other_features(data_head, parse_lst_path, config, path_head, context=context, parse=parse)
        _    = save_other_features(data_tail, parse_lst_path, config, path_tail, context=context, parse=parse)
        multi_compute(data, path, path_head, path_tail, omit)
        return
    # based on aspect word
    data['word_len'] = data['token'].str.len().to_numpy()
    data['num_syllables'] = data['token'].apply(lambda x: syllables.estimate(str(x))).to_numpy()
    data['num_hyponyms'] = data.apply(lambda x: len(get_hyponyms(x['sentence'] if context else None, x['token'], disambiguate=config['disambiguate'] if context else False)), axis=1).to_numpy()
    data['num_hypernyms'] = data.apply(lambda x: len(get_hypernyms(x['sentence'] if context else None, x['token'], disambiguate=config['disambiguate'] if context else False)), axis=1).to_numpy()
    data['is_acronym'] = (data['token'].str.isupper()*1).to_numpy()
    data['is_pronoun'] = (data['token'].apply(lambda x: x[0].isupper())*1).to_numpy()
    # based on context
    omit = set()
    if context:
        corpus_dummies = pd.get_dummies(data['corpus'], prefix='corpus')
        for corpus_name in corpus_dummies:
            data[corpus_name] = corpus_dummies[corpus_name]
            omit.add(corpus_name)
        tagdict = load('help/tagsets/upenn_tagset.pickle')
        tags = [tag for tag in tagdict.keys() if tag[0] not in punctuation]
        POS = data.apply(lambda x: get_POS(x['sentence'], x['token']), axis=1)
        for tag in tags:
            data['POS_'+tag] = (POS == tag) * 1
        funcs = ["textstat." + func[0] for func in inspect.getmembers(textstat, predicate=inspect.ismethod)]
        for elem in tqdm(funcs):
            method = eval(elem)
            if method.__name__ in ['difficult_words_list', 'set_lang', 'text_standard', 'dale_chall_readability_score_v2', 'dale_chall_readability_score', 'gunning_fog', 'spache_readability', 'avg_sentence_length', 'avg_sentence_per_word', 'sentence_count', 'difficult_words', 'is_difficult_word', 'is_easy_word', 'smog_index']:
                continue
            textstat.set_lang("en")
            data[method.__name__] = data['sentence'].apply(lambda x: method(x)).to_numpy()
            omit.add(method.__name__)
        data['SMOGIndex'] = data['sentence'].apply(lambda x: readability.getmeasures(x, lang='en')['readability grades']['SMOGIndex']).to_numpy()
        data['DaleChallIndex'] = data['sentence'].apply(lambda x: readability.getmeasures(x, lang='en')['readability grades']['DaleChallIndex']).to_numpy()
        omit.add('SMOGIndex'); omit.add('DaleChallIndex')
        if parse:
            parse_lst = pkl.load(open(parse_lst_path, 'rb'))
            parse_tree_depths = []
            token_depths = []
            num_words_at_depths = []
            for parse_tree, token in tqdm(zip(parse_lst, data['token'])):
                parse_tree_depths.append(parse_tree.height())
                token_depths.append(token_depth(parse_tree, token))
                num_words_at_depths.append(num_words_at_depth(parse_tree, token_depths[-1]))
            data['parse_tree_depth'] = np.array(parse_tree_depths).astype(np.int64)
            omit.add('parse_tree_depth')
            data['token_depth'] = np.array(token_depths).astype(np.int64)
            data['num_words_at_depth'] = np.array(num_words_at_depths).astype(np.int64)
    data.to_csv(path, sep='\t')
    return omit

conversion_dict = {'‚Ä†':'†', '¬∞':'°', '¬¢':'¢', '¬£':'£', '¬ß':'§', '‚Ä¢':'•', '¬∂':'¶', '¬Æ':'®', '¬©':'©', '‚Ñ¢':'™', '¬¥':'´', '¬®':'¨', '‚â†':'≠', '‚àû':'∞', '¬±':'±', '‚â§':'≤', '‚â•':'≥', '¬•':'¥', '‚àÇ':'∂', '‚àë':'∑', '‚àè':'∏', '‚à´':'∫', '¬ø':'¿', '¬°':'¡', '¬¨':'¬', '‚àö':'√', '‚âà':'≈', '‚àÜ':'∆', '¬´':'«', '¬ª':'»', '‚Ä¶':'…', '‚Äì':'–', '‚Äî':'—', '‚Äú':'“', '‚Äù':'”', '‚Äò':'‘', '‚Äô':'’', '√∑':'÷', '‚óä':'◊', '‚ÅÑ':'⁄', '‚Ç¨':'€', '‚Äπ':'‹', '‚Ä∫':'›', '‚Ä°':'‡', '¬∑':'·', '‚Äö':'‚', '‚Äû':'„', '‚Ä∞':'‰', 'ÀÜ':'ˆ', 'Àú':'˜', '¬Ø':'¯', 'Àò':'˘', 'Àô':'˙', 'Àö':'˚', '¬∏':'¸', 'Àù':'˝', 'Àõ':'˛', 'Àá':'ˇ', '¬™':'ª', '√°':'á', '√Å':'Á', '√†':'à', '√Ä':'À', '√¢':'â', '√Ç':'Â', '√Ö':'Å', '√•':'å', '√Ñ':'Ä', '√§':'ä', '√£':'ã', '√É':'Ã', '√Ü':'Æ', '√¶':'æ', '√á':'Ç', '√ß':'ç', '√â':'É', '√©':'é', '√®':'è', '√à':'È', '√™':'ê', '√ä':'Ê', '√´':'ë', '√ã':'Ë', 'Ô¨Å':'ﬁ', 'Ô¨Ç':'ﬂ', '∆í':'ƒ', '√≠':'í', '√ç':'Í', '√¨':'ì', '√å':'Ì', '√Æ':'î', '√é':'Î', '√Ø':'ï', '√è':'Ï', 'ƒ±':'ı', '√ë':'Ñ', '√±':'ñ', '¬∫':'º', '√≥':'ó', '√ì':'Ó', '√≤':'ò', '√í':'Ò', '√¥':'ô', '√î':'Ô', '√ñ':'Ö', '√∂':'ö', '√µ':'õ', '√ï':'Õ', '≈í':'Œ', '≈ì':'œ', '√ò':'Ø', '√∏':'ø', '√ü':'ß', '√∫':'ú', '√ö':'Ú', '√π':'ù', '√ô':'Ù', '√ª':'û', '√õ':'Û', '√ú':'Ü', '√º':'ü', '√ø':'ÿ', '≈∏':'Ÿ', '¬µ':'µ', 'œÄ':'π', 'Œ©':'Ω', 'Œ±':'α', '     ':'', '‚àí':'−', 'Œ≤':'β', 'Œî':'Δ', '√ó':'×', 'Œ¥':'δ', 'Œº':'μ', 'for   CCAAT':'forCCAAT', '(‚“':'(“', '‚”)':'”)', 'Œ≥':'γ'}

def clean_sentence(sentence):
    result = sentence
    for char_old, char_new in conversion_dict.items():
        result = result.replace(char_old, char_new)
    # this duplicate is not a type-o :)
    for char_old, char_new in conversion_dict.items():
        result = result.replace(char_old, char_new)
    result = result.replace('%', '%25')
    return result

def log_transform(data, columns, path):
    if columns == 'all':
        for column in data:
            if column in ['id','corpus','sentence','token','complexity','class']:
                continue
            feature = data[column]
            if feature.dtype.type == np.object_:
                continue
            data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    elif columns in data:
        feature = data[columns]
        data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    elif isinstance(columns, list):
        for column in columns:
            feature = data[column]
            data['log_{}'.format(column)] = np.log(feature-(min(feature)-1))
    data.to_csv(path, sep='\t')

def preprocess_for_bert(data, job, name, do_round=True, do_mask=False, do_scale=False, do_token=True, do_split_token=False):
    output_path = './data/'+job+'/'+name+'_bert.tsv'
    if 'complexity' not in data:
        data.insert(0, 'complexity', 0.0)
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
    if do_split_token:
        x_copy = x.copy()
        x_copy['token'] = x_copy['token'].apply(lambda x: x.split(' ')[0])
        output_head_path = './data/'+job+'/'+name+'_head_bert.tsv'
        x_copy.to_csv(output_head_path, sep='\t', index=False, header=False, encoding='utf-8')
        x_copy = x.copy()
        x_copy['token'] = x_copy['token'].apply(lambda x: x.split(' ')[1])
        output_tail_path = './data/'+job+'/'+name+'_tail_bert.tsv'
        x_copy.to_csv(output_tail_path, sep='\t', index=False, header=False, encoding='utf-8')

def custom_cv_folds(data, n_splits=5, test_size=0.1):
    std_idx = range(len(data))
    sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    for train_index, dev_index in sss.split(std_idx, data[['corpus', 'class']]):
        yield train_index, dev_index

def multi_data(data):
    data_head = data.copy()
    data_head['token'] = data_head['token'].apply(lambda x: x.split(' ')[0])
    data_tail = data.copy()
    data_tail['token'] = data_tail['token'].apply(lambda x: x.split(' ')[1])
    return data_head, data_tail

def multi_compute(data, path, path_head, path_tail, omit=set()):
    columns_with_lists = ['glove_word', 'glove_context', 'infersent_embeddings', 'elmo_word']
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
        
    data.to_csv(path, sep='\t')

def MAE(y, y_pred):
    return mean_absolute_error(y, y_pred)

def pearson(y, y_pred):
    return pearsonr(y, y_pred)[0]

def spearman(y, y_pred):
    return spearmanr(y, y_pred)[0]

def get_ngrams(sentence, token, N):
    return [' '.join(ngram) for ngram in ngrams(word_tokenize(sentence), N) if token in ngram]

def get_tf(word, exact=False):
    encoded_query = urllib.parse.quote(word) 
    params = {'corpus': 'eng-us', 'query': encoded_query, 'format': 'json'} 
    params = '&'.join('{}={}'.format(name, value) for name, value in params.items()) 
    response = requests.get('https://api.phrasefinder.io/search?' + params) 
    response = json.loads(response.text)
    if 'phrases' not in response:
        return 0
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

def get_sorted_mi(path):
    mi = []
    with open(path, 'r') as file:
        for line in file:
            f, m = line.strip('\n').split(', ')
            m = float(m)
            mi.append((f, m))
    mi = sorted([(f, m) for (f, m) in mi], key=lambda x: x[1], reverse=True)
    return map(list, zip(*mi))