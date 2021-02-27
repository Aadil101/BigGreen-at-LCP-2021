import pickle5 as pkl
from utils import *
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import numpy as np
import pandas as pd
import ast
import json

def single_picks_2_multi_picks(single_picks, single_feature_names, multi_feature_names, mode='head'):
    multi_picks = []
    single_feature_names_set = set(single_feature_names)
    multi_feature_names_set = set(multi_feature_names)
    for single_pick in single_picks:
        if single_pick.startswith('glove_word') or single_pick.startswith('elmo_word'):
            num = int(single_pick[single_pick.rindex('_')+1:])
            single_pick_wout_num = single_pick.strip('_{}'.format(num))
            if mode == 'head':
                multi_picks.append('{}_head_{}'.format(single_pick_wout_num, num))
            if mode == 'tail':
                multi_picks.append('{}_tail_{}'.format(single_pick_wout_num, num))
        elif single_pick.startswith('infersent_embeddings') or single_pick.startswith('glove_context'):
            multi_picks.append(single_pick)
        elif '{}_head'.format(single_pick) in multi_feature_names:
            if mode == 'head':
                multi_picks.append('{}_head'.format(single_pick))
            if mode == 'tail':
                multi_picks.append('{}_tail'.format(single_pick))
        elif single_pick in multi_feature_names:
            multi_picks.append(single_pick)
        else:
            print('foo')
    return multi_picks

job = 'pickle'
name = 'multi_train'
path = './data/'+job+'/'+name+'_j.tsv'
print('loading data and features...')
data = pd.read_csv(path, sep='\t', index_col=0)
features_stacked = load_features_stacked(job, name, data)
complexity = data['complexity'].to_numpy()
discrete_features = discrete(data)
feature_names = names(data)
feature_name_2_idx = load_feature_name_2_idx(job, name, data)

print(len(data.columns))
print(len(feature_names))
print('...done')

print('mutual info regression...')
mi = mutual_info_regression(features_stacked, complexity, discrete_features=discrete_features)
print('...done')

print('saving...')
with open('./data/'+job+'/{}_mi.txt'.format(name), 'w') as file:
    for f, m in zip(feature_names, mi):
        file.write('{}, {}\n'.format(f, m))
print('...boom!')

'''
print('applying PCA...')
n_components = 0.95
scaler_gw, pca_gw, features['glove_word_pca'] = apply_pca(features['glove_word'], n_components)
scaler_gs, pca_gs, features['glove_context_pca'] = apply_pca(features['glove_context'], n_components)
scaler_ie, pca_ie, features['infersent_embeddings_pca'] = apply_pca(features['infersent_embeddings'], n_components)
print('...done')

print('stuff...')
glove_word = features['glove_word']
glove_context = features['glove_context']
infersent_embeddings = features['infersent_embeddings']
del features['glove_word']
del features['glove_context']
del features['infersent_embeddings']

features_stacked = np.hstack([item[:,None] if item.ndim == 1 else item for item in features.values()])
complexity = data['complexity'].to_numpy()
discrete_features = discrete(features)
print('...done')

print('mutual info regression...')
mi = mutual_info_regression(features_stacked, complexity, discrete_features=discrete_features)
print('...done')
print('saving...')
with open('feature_selection/mi.pkl', 'wb') as file:
    pkl.dump(mi, file)
with open('feature_selection/feature_names.pkl', 'wb') as file:
    pkl.dump(names(features), file)
print('...boom!')
'''
