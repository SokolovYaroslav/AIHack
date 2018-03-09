
# coding: utf-8
import pandas as pd
import numpy as np
from datetime import timedelta
import os

def load_data(resave=False):
    """
    If data.hdf doesn't exist, creates it (from train_data.csv and test_data.csv),
    dropping from train & test 'Unnamed: 0' and transforming date to datetime
    If exists, just loads it
    
    resave (bool): whether to respawn data.hdf again
    """
    if not os.path.isfile('data/data.hdf') or resave:
        train = pd.read_csv('data/train_data.csv').drop('Unnamed: 0', axis=1)
        test = pd.read_csv('data/test_data.csv').drop('Unnamed: 0', axis=1)
        train['date'] = pd.to_datetime(train.date)
        test['date'] = pd.to_datetime(test.date)
        train.to_hdf('data/data.hdf', 'train')
        test.to_hdf('data/data.hdf', 'test')
    return pd.read_hdf('data/data.hdf', 'train'), pd.read_hdf('data/data.hdf', 'test')


def calculate_target(train, offset=0):
    """
    Returns X_train without last 30 days and Series with index=ids, values=target
    Target is built only for users who were present in the X_train (w\o last 30 days)
    
    offset (int): month for target is chosen as train.month.max() - offset
    """
    target_month = train.date.dt.month.max() - offset
    X_train = train.loc[train.date.dt.month < target_month]
    
    users = train.loc[train.date.dt.month == target_month].id.unique()
    #or aggregate by sum_b, see if the same
    users = np.intersect1d(users, X_train.id.unique())
    
    target = pd.Series(np.zeros((X_train.id.nunique())), index=X_train.id.unique())
    target.loc[users] = 1
    return X_train, target

def train_test_split(X_train, y_train, train_size=0.75):
    if train_size < 1:
        train_size = int(X_train.id.nunique() * train_size)
    assert(X_train.id.nunique() == y_train.shape[0])
    split_id = X_train.id.unique()[train_size]
    split_index = np.where(X_train.id == split_id)[0].min()
    return X_train.iloc[:split_index, :], X_train.iloc[split_index:, :],           y_train.iloc[:train_size], y_train.iloc[train_size:]


# Example usage:

#X_train, y_train = calculate_target(train, offset=0)
#X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train)

