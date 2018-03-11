
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import timedelta
from datetime import datetime
from sklearn.metrics import roc_auc_score
from scipy import stats
import os
import math

def log(x):
    if x < 0:
        return -log(-x)
    else: 
        return math.log(x + 1)

# In[2]:


def load_data(resave=False, points=True, sort=False):
    """
    If data.hdf doesn't exist, creates it (from train_data.csv and test_data.csv),
    dropping from train & test 'Unnamed: 0' and transforming date to datetime
    If exists, just loads it
    
    resave (bool): whether to respawn data.hdf again
    """
    if points:
        train_path = 'data/train_points.csv'
        test_path = 'data/test_points.csv'
    else:
        train_path = 'data/train_data.csv'
        test_path = 'data/test_data.csv'

    if not os.path.isfile('data/data.hdf') or resave:
        train = pd.read_csv(train_path, low_memory=False)
        test = pd.read_csv(test_path, low_memory=False)
        if sort:
            train = train.sort_values(by=['id', 'date'])
            test = test.sort_values(by=['id', 'date'])
        if 'Unnamed: 0' in train.columns:
            train = train.drop('Unnamed: 0', axis=1)
            test = test.drop('Unnamed: 0', axis=1)
        train['date'] = pd.to_datetime(train.date)
        test['date'] = pd.to_datetime(test.date)
        train['first_prch'] = train.first_prch.apply(lambda x: datetime.strptime(x, '%d.%m.%y %H:%M:%S'))
        test['first_prch'] = test.first_prch.apply(lambda x: datetime.strptime(x, '%d.%m.%y %H:%M:%S'))
        train.to_hdf('data/data.hdf', 'train')
        test.to_hdf('data/data.hdf', 'test')
    return pd.read_hdf('data/data.hdf', 'train'), pd.read_hdf('data/data.hdf', 'test')

def save_to(train, test, path):
    train.to_hdf(path, 'train')
    test.to_hdf(path, 'test')
    
def load_from(path):
    return pd.read_hdf(path, 'train'), pd.read_hdf(path, 'test')

def get_cat_features():
    return ['n_tr', 'code_azs', 'location', 'region', 'code',
           'code1', 'type', 'month', 'weekday']

def add_features(train, test, triang=False, rolling_window=[], sort=False):
    # Number of first dates of a user
    train_num_frst_purch = train[['id', 'first_prch']].groupby('id').first_prch.nunique()
    train_num_frst_purch = pd.DataFrame(train_num_frst_purch).reset_index()
    train = train.merge(train_num_frst_purch, left_on='id', right_on='id', how='outer')
    train = train.rename(index=str, columns={"first_prch_x": "first_prch", "first_prch_y": "first_prch_num"})
    test_num_frst_purch = test[['id', 'first_prch']].groupby('id').first_prch.nunique()
    test_num_frst_purch = pd.DataFrame(test_num_frst_purch).reset_index()
    test = test.merge(test_num_frst_purch, left_on='id', right_on='id', how='outer')
    test = test.rename(index=str, columns={"first_prch_x": "first_prch", "first_prch_y": "first_prch_num"})
    # reaches by 4 category: 25%, 50%, 75%
    #spend_by_users = train.groupby('id')['sum_b'].sum()
    #q25, q50, q75 = [i for i in spend_by_users.describe()[['25%', '50%', '75%']]]
    #train['total_user_spend'] = train['id'].apply(lambda x: spend_by_users[x])
    #train['rich_category'] = train['total_user_spend'].apply(lambda x: get_rich_category(x, q25, q50, q75))
    
    #spend_by_users = test.groupby('id')['sum_b'].sum()
    #q25, q50, q75 = [i for i in spend_by_users.describe()[['25%', '50%', '75%']]]
    #test['total_user_spend'] = test['id'].apply(lambda x: spend_by_users[x])
    #test['rich_category'] = test['total_user_spend'].apply(lambda x: get_rich_category(x, q25, q50, q75))
    # count returns of product
    #train_neg = train[train['sum_b'] < 0]
    #train_neg_count = train_neg[['id']].groupby('id').size().reset_index(name='return_num')
    #train = train.merge(train_neg_count, left_on='id', right_on='id', how='outer')
    #train['return_num'].fillna(0, inplace=True)
    #test_neg = test[test['sum_b'] < 0]
    #test_neg_count = test_neg[['id']].groupby('id').size().reset_index(name='return_num')
    #test = test.merge(test_neg_count, left_on='id', right_on='id', how='outer')
    #test['return_num'].fillna(0, inplace=True)
    # replace all first_prch with earliest first_prch
    #train_first = train.groupby('id').first_prch.min().reset_index(name='first_prch')
    #train = train.drop('first_prch', axis=1).merge(train_first, left_on='id', right_on='id', how='outer')
    #test_first = test.groupby('id').first_prch.min().reset_index(name='first_prch')
    #test = test.drop('first_prch', axis=1).merge(test_first, left_on='id', right_on='id', how='outer')
    # replace all first_prch with earliest first_prch
    #train_first = train.groupby('id').first_prch.min().reset_index(name='first_prch')
    #train = train.drop('first_prch', axis=1).merge(train_first, left_on='id', right_on='id', how='outer')
    #test_first = test.groupby('id').first_prch.min().reset_index(name='first_prch')
    #test = test.drop('first_prch', axis=1).merge(test_first, left_on='id', right_on='id', how='outer')
    # mean oil price for every oil type code
    #train_no_q = train[(train['q'] == 0) & (train['sum_b'] > 0)]
    #train_no_q['oil_price'] = train_no_q['sum_b'] / train_no_q['v_l']
    #train_no_q_group = train_no_q[['code','oil_price']].groupby('code').agg('mean').reset_index()
    #train = train.merge(train_no_q_group, left_on='code', right_on='code', how='outer')
    #train['oil_price'].fillna(0, inplace=True)
    #train['oil_price'] = train['oil_price'].replace(np.inf, 0)
    #test_no_q = test[(test['q'] == 0) & (test['sum_b'] > 0)]
    #test_no_q['oil_price'] = test_no_q['sum_b'] / test_no_q['v_l']
    #test_no_q_group = test_no_q[['code','oil_price']].groupby('code').agg('mean').reset_index()
    #test = test.merge(test_no_q_group, left_on='code', right_on='code', how='outer')
    #test['oil_price'].fillna(0, inplace=True)
    #test['oil_price'] = test['oil_price'].replace(np.inf, 0)
    # spend_on_fuel
    #x = pd.DataFrame(np.array([train['id'], train['v_l'] * train['oil_price']]).T)
    #spend_on_fuel = x.groupby(0)[1].sum()
    #train['user_spend_fuel'] = train['id'].apply(lambda x: spend_on_fuel[x])
    
    #x = pd.DataFrame(np.array([test['id'], test['v_l'] * test['oil_price']]).T)
    #spend_on_fuel = x.groupby(0)[1].sum()
    #test['user_spend_fuel'] = test['id'].apply(lambda x: spend_on_fuel[x])
    #if triang:
    # triang 8 windows
        #train_roll_mean = train[['date', 'sum_b']].set_index('date')
        #train_roll_mean_triang_8 = train_roll_mean.rolling(8, win_type='triang').mean().rename(index=str, columns={"sum_b": "roll_win_triang_8"})
        #train = train.append(train_roll_mean_triang_8, ignore_index=True, axis=1)
        #test_roll_mean = test[['date', 'sum_b']].set_index('date')
        #test_roll_mean_triang_8 = test_roll_mean.rolling(8, win_type='triang').mean().rename(index=str, columns={"sum_b": "roll_win_triang_8"})
        #test = test.append(test_roll_mean_triang_8, ignore_index=True, axis=1)

#     # TOO MUCH MEMORY USED, UNCOMMENT WHEN CERTAIN 
#     # bartlett 8 windows
#     train_roll_mean = train[['date', 'sum_b']].set_index('date')
#     train_roll_mean_bartlett_8 = train_roll_mean.rolling(16, win_type='bartlett').mean().rename(index=str, columns={"sum_b": "roll_win_bartlett_8"})
#     train = train.append(train_roll_mean_bartlett_8, ignore_index=True)
#     test_roll_mean = test[['date', 'sum_b']].set_index('date')
#     test_roll_mean_bartlett_8 = test_roll_mean.rolling(16, win_type='bartlett').mean().rename(index=str, columns={"sum_b": "roll_win_bartlett_8"})

#     test = test.append(test_roll_mean_bartlett_8, ignore_index=True)
#     # bartlett 16 windows
#     train_roll_mean = train[['date', 'sum_b']].set_index('date')
#     train_roll_mean_bartlett_16 = train_roll_mean.rolling(16, win_type='bartlett').mean().rename(index=str, columns={"sum_b": "roll_win_bartlett_16"})
#     train = train.append(train_roll_mean_bartlett_16, ignore_index=True)
#     test_roll_mean = test[['date', 'sum_b']].set_index('date')
#     test_roll_mean_bartlett_16 = test_roll_mean.rolling(16, win_type='bartlett').mean().rename(index=str, columns={"sum_b": "roll_win_bartlett_16"})
#     test = test.append(test_roll_mean_bartlett_16, ignore_index=True

#     # blackmanharris 16 windows
#     train_roll_mean = train[['date', 'sum_b']].set_index('date')
#     train_roll_mean_blackmanharris_16 = train_roll_mean.rolling(16, win_type='blackmanharris').mean().rename(index=str, columns={"sum_b": "roll_win_blackmanharris_16"})
#     train = train.append(train_roll_mean_blackmanharris_16, ignore_index=True)
#     test_roll_mean = test[['date', 'sum_b']].set_index('date')
#     test_roll_mean_blackmanharris_16 = test_roll_mean.rolling(16, win_type='blackmanharris').mean().rename(index=str, columns={"sum_b": "roll_win_blackmanharris_16"})
#     test = test.append(test_roll_mean_blackmanharris_16, ignore_index=True)

#     # triang 16 windows
#     train_roll_mean = train[['date', 'sum_b']].set_index('date')
#     train_roll_mean_triang_16 = train_roll_mean.rolling(16, win_type='triang').mean().rename(index=str, columns={"sum_b": "roll_win_triang_16"})
#     train = train.append(train_roll_mean_triang_16, ignore_index=True)
#     test_roll_mean = test[['date', 'sum_b']].set_index('date')
#     test_roll_mean_triang_16 = test_roll_mean.rolling(16, win_type='triang').mean().rename(index=str, columns={"sum_b": "roll_win_triang_16"})
#     test = test.append(test_roll_mean_triang_16, ignore_index=True)

#     # blackmanharris 8 windows
#     train_roll_mean = train[['date', 'sum_b']].set_index('date')
#     train_roll_mean_blackmanharris_8 = train_roll_mean.rolling(8, win_type='blackmanharris').mean().rename(index=str, columns={"sum_b": "roll_win_blackmanharris_8"})
#     train = train.append(train_roll_mean_blackmanharris_8, ignore_index=True)
#     test_roll_mean = test[['date', 'sum_b']].set_index('date')
#     test_roll_mean_blackmanharris_8 = test_roll_mean.rolling(8, win_type='blackmanharris').mean().rename(index=str, columns={"sum_b": "roll_win_blackmanharris_8"})
#     test = test.append(test_roll_mean_blackmanharris_8, ignore_index=True)

    # time features
    train['month'] = train.date.dt.month
    #train['weekday'] = train.date.dt.dayofweek
    test['month'] = test.date.dt.month
    #test['weekday'] = test.date.dt.dayofweek
    #train['full_month'] = (train.date.dt.year - 2016)*12 + train.date.dt.month
    #test['full_month'] = (test.date.dt.year - 2016)*12 + test.date.dt.month
    train['days'] = (train.date.dt.year-2016)*365+train.date.dt.dayofyear
    test['days'] = (test.date.dt.year-2016)*365+test.date.dt.dayofyear
    train['time_weight'] = 1 / train.days
    test['time_weight'] = 1 / test.days
    #train['hour'] = -9999
    #train.loc[~train.time.isnull(), 'hour'] = train.loc[~train.time.isnull(), 'time']\
                                          #.apply(lambda x: x.split(':')[0].strip()).astype(int)
    #test['hour'] = -9999
    #test.loc[~test.time.isnull(), 'hour'] = test.loc[~test.time.isnull(), 'time']\
                                          #.apply(lambda x: x.split(':')[0].strip()).astype(int)
    #max_day = train.days.max()
    #days_last = train.groupby('id')['days'].apply(lambda df: max_day - df.iloc[-1])
    #train = train.merge(days_last.reset_index().rename(columns={'days': 'days_since_last'}), on='id')
    #days_last = test.groupby('id')['days'].apply(lambda df: max_day - df.iloc[-1])
    #test = test.merge(days_last.reset_index().rename(columns={'days': 'days_since_last'}), on='id')
    # true percent
    #train['tmp'] = train['cur_points']
    #train.tmp = train.tmp.apply(lambda x: x if x < 0 else 0)
    #train = train.merge(train.groupby('id').tmp.min().reset_index(name='min_point_bal'),
    #           left_on='id', right_on='id', how='outer')
    #train['cur_points'] = train.loc[:, 'cur_points'] - train.loc[:, 'min_point_bal']
    #train['tmp'] = train.loc[:, 'cur_points'] - train.loc[:, 'percent']
    #train.tmp = train.tmp.apply(lambda x: x if x < 0 else 0)
    #train = train.merge(train.groupby('id').tmp.min().reset_index(name='tmp_1'),
    #                    left_on='id', right_on='id', how='outer')
    #train['cur_points'] = train.loc[:, 'cur_points'] - train.loc[:, 'tmp_1']
    #train['true_percent'] = ((train.loc[:,'percent'] / train.loc[:,'cur_points']) * 100).fillna(0)
    #train.drop(['tmp', 'tmp_1', 'min_point_bal'], axis=1, inplace=True)
    
    #test['tmp'] = test['cur_points']
    #test.tmp = test.tmp.apply(lambda x: x if x < 0 else 0)
    #test = test.merge(test.groupby('id').tmp.min().reset_index(name='min_point_bal'),
    #            left_on='id', right_on='id', how='outer')
    #test['cur_points'] = test.loc[:, 'cur_points'] - test.loc[:, 'min_point_bal']
    #test['tmp'] = test.loc[:, 'cur_points'] - test.loc[:, 'percent']
    #test.tmp = test.tmp.apply(lambda x: x if x < 0 else 0)
    #test = test.merge(test.groupby('id').tmp.min().reset_index(name='tmp_1'),
    #                    left_on='id', right_on='id', how='outer')
    #test['cur_points'] = test.loc[:, 'cur_points'] - test.loc[:, 'tmp_1']
    #test['true_percent'] = ((test.loc[:,'percent'] / test.loc[:,'cur_points']) * 100).fillna(0)
    #test.drop(['tmp', 'tmp_1', 'min_point_bal'], axis=1, inplace=True)
    # logarithmic values
    train.sum_b = train.sum_b.apply(log)
    test.sum_b = test.sum_b.apply(log)
    train.q = train.q.apply(log)
    test.q = test.q.apply(log)
    train.v_l = train.v_l.apply(log)
    test.v_l = test.v_l.apply(log)
    #train.total_user_spend = train.total_user_spend.apply(log)
    #test.total_user_spend = test.total_user_spend.apply(log)
    #train.user_spend_fuel = train.user_spend_fuel.apply(log)
    #test.user_spend_fuel = test.user_spend_fuel.apply(log)
    
    if sort:
        train = train.sort_values(by=['id', 'date'])
        test = test.sort_values(by=['id', 'date'])
    return train, test


def calculate_target(train, offset=0, sort_y=True, by_sum_b=False):
    """
    Returns X_train without last 30 days and Series with index=ids, values=target
    Target is built only for users who were present in the X_train (w\o last 30 days)
    
    offset (int): month for target is chosen as train.month.max() - offset
    """
    target_month = train.date.dt.month.max() - offset
    X_train = train.loc[train.date.dt.month < target_month]
    
    dates = train.date.dt 
    train['full_month'] = (dates.year - 2016)*12 + dates.month
    
    target_month = train.full_month.max() - offset
    X_train = train.loc[train.full_month < target_month]
    users_last_month = X_train.loc[X_train.full_month == target_month - 1].id.unique()
    X_train = X_train.set_index('id').loc[users_last_month].reset_index()
    #X_train = X_train.loc[X_train.loc[:, 'id'].apply(lambda x: x in users_last_month), :]
    
    if by_sum_b:
        users = train.loc[train.full_month == target_month].groupby('id')['sum_b']\
               .apply(lambda gr: gr.fillna(gr.mean()).sum())
    else:
        users = train.loc[train.full_month == target_month].id.unique()
    
    users = np.intersect1d(users, X_train.id.unique())
    
    target = pd.Series(np.ones((X_train.id.nunique())), index=X_train.id.unique())
    target.loc[users] = 0
    if sort_y:
        target = target.sort_index()
        
    return X_train, target

def train_test_split(X_train, y_train, train_size=0.75):
    if train_size < 1:
        train_size = int(X_train.id.nunique() * train_size)
    assert(X_train.id.nunique() == y_train.shape[0])
    split_id = X_train.id.unique()[train_size]
    split_index = np.where(X_train.id == split_id)[0].min()
    return X_train.iloc[:split_index, :], X_train.iloc[split_index:, :],\
            y_train.iloc[:train_size], y_train.iloc[train_size:]

def get_rich_category(user_spend, q25, q50, q75):
    if user_spend < q25:
        return 0
    elif user_spend < q50:
        return 1
    elif user_spend < q75:
        return 2
    else:
        return 3
    
def save_split(file, X_tr, X_val, y_tr, y_val,num):
    num = str(num)
    X_tr.to_hdf(file, 'X_tr'+num)
    X_val.to_hdf(file, 'X_val'+num)
    y_tr.to_hdf(file, 'y_tr'+num)
    y_val.to_hdf(file, 'y_val'+num)
    
    
def load_split(file, num):
    num = str(num)
    return pd.read_hdf(file, 'X_tr'+num), pd.read_hdf(file, 'X_val'+num), \
            pd.read_hdf(file, 'y_tr'+num), pd.read_hdf(file, 'y_val'+num)

def cross_val(clf, X_train, aggregate_func, return_proba=False,
              splits=3, interval=0, train_size=0.75, verbose=True, splits_file=None,
             cat_features=None):
    """
    Makes a few splits, in each of them makes train_test_split with a new offset,
    then applies aggregate_func to X_tr and X_val. Trains clf on (X_tr, y_tr),
    counts roc_auc_score and appends it to scores.
    
    aggregate_func (func): takes X dataset and aggregates by id. Should return 
                           either DF with non-multi index, or numpy 2D-array
    return_proba (bool): if True, returns (scores, probas)
                         if False, returns only scores
    """
    scores = []
    probas = []
    for split_ind in range(splits):
        if verbose:
            print("Split №", split_ind)
        
        offset = split_ind*(1+interval)
        if splits_file == None:
            X, y = calculate_target(X_train, offset=offset)
            X_tr, X_val, y_tr, y_val = train_test_split(X, y, train_size=train_size)
            if verbose:
                print("Adding features...")
            X_tr, X_val = add_features(X_tr, X_val, sort=True)
            if verbose:
                print("Aggregating X_tr..")
            X_tr = aggregate_func(X_tr, take_values=False)
            if verbose:
                print("Aggregating X_val..")
            X_val = aggregate_func(X_val, take_values=False)
        else:
            X_tr, X_val, y_tr, y_val = load_split(splits_file, offset)
        
        if verbose:
            print("Fitting classifier..")
        
        if cat_features is not None:
            clf.fit(X_tr.values, y_tr, cat_features=cat_features)
        else:
            clf.fit(X_tr.values, y_tr)
            
        pred = clf.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, pred))
        if verbose:
            print("Target month: -{}. Score: {}".format(offset, scores[-1]))
            print("-------------------")
        probas.append(pd.Series(pred, index=X_val.index))
    if verbose:
        print("Mean score is:", np.mean(scores))
    if return_proba:
        return scores, probas
    return scores

def unique_cnt(series):
    return series.unique().shape[0]

mode_func = lambda x: stats.mode(x).mode[0]

def aggregate(df, take_values=True):
    mode = lambda x: stats.mode(x).mode[0]
    num_features = ['min', 'max', 'mean', 'median', 'sum']
    cat_features = [unique_cnt, mode]
    
    res = df.groupby('id')[['v_l', 'q', 'sum_b', 'location', 'code', 'percent', 'type', 'month',\
                            'weekday', 'code_azs','region', 'code1', 'oil_price', 'cur_points']].agg({
    'v_l':num_features, 'q':num_features, 'sum_b':num_features, 'percent':num_features, 
    'location':cat_features,'type':cat_features, 'month':cat_features, 'weekday':cat_features,
    'code_azs':cat_features,'month':cat_features, 'weekday':cat_features,'region':cat_features,
    'code1':cat_features, 'oil_price':cat_features, 'cur_points':cat_features,
    'code':[unique_cnt],
})
    if take_values:
        return res.values, res.index
    else:
        return res

# Example usage:

#X_train, y_train = calculate_target(train, offset=0)

#X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

