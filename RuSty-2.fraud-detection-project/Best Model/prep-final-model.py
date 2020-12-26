#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
!pip install heatmapz
from heatmap import heatmap, corrplot


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import gc
from hyperopt import hp, fmin, tpe, Trials
from sklearn.metrics import roc_auc_score
from functools import partial
from hyperopt.pyll.base import scope
import xgboost as xgb
from xgboost import cv
from sklearn.metrics import roc_auc_score
from xgboost import plot_importance
import matplotlib.pyplot as plt
import time
import datetime
from contextlib import contextmanager
from lightgbm import LGBMClassifier
from sklearn import metrics, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA 
import missingno as msno
from sklearn.preprocessing import LabelEncoder




pd.set_option('display.max_rows', 999)
pd.set_option('display.max_columns',700)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

gc.collect()
# # %% [markdown]
# Had 339 V columns with no idea what they were, and most of them had more than 50% null values which it difficult to decide what to do with them. EDA didn't make much sense with the amount of null values we had. So we grouped them based on the similar number of null values they had, and checked for correlation. Removed the V columns which had high correlation with each other and then moved forward with the remaining columns. Some V columns had correlation as high as >0.8 which is pretty big so this technique helped.

# %% [code]
txn = pd.read_csv('../input/ieee-fraud/train_transaction.csv', index_col='TransactionID')
nans = txn.isna().sum()


# %% [code]
nans_dict = {}

for row in nans.iteritems():
    try:
        nans_dict[str(row[1])].append( row[0] )
    except:
        nans_dict[str(row[1])] = [row[0]]

# %% [code]
print( sorted([int(i) for i in nans_dict.keys()]) )

# %% [code]
for key, val in nans_dict.items():
    print(f"Number of NULLS: {key}\n {val}\n")

# %% [code]
nans_groups = []
for l in nans_dict.values():
    if len(l) > 3:
        nans_groups.append(l)
nans_groups

# %% [code]
def show_corr(lst):
#     plt.figure(figsize=(15,15))
    sns.set( font_scale = 1 )
    sns.heatmap( txn[lst].corr(), cmap='BrBG', annot=True, fmt='0.2f', square=True, linewidths = 2, vmin = -1, vmax = 1 )
#     pass

# %% [code]
final_cols = []

# %% [code]
def max_value_counts(lst, final_cols):
    final = []
    for g in lst:
        gval = [len(txn[str(gg)].value_counts()) for gg in g]
        most_features_index = np.argmax(gval)
        final.append( g[most_features_index] )
    final_cols += final

# %% [code]
plt.figure( figsize=(15, 15) )
show_corr(nans_groups[1])

# %% [code]
gcorr = [['D1'], ['V281'], ['V282'], ['V283'], ['V288', 'V289'], ['V296'], ['V300', 'V301'], ['V313', 'V314', 'V315']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(15, 15) )
show_corr(nans_groups[2])

# %% [code]
gcorr = [['D11'], ['V1'], ['V2', 'V3'], ['V4', 'V5'], ['V6', 'V7'], ['V8', 'V9'], ['V10', 'V11']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(20, 20) )
show_corr(nans_groups[3])

# %% [code]
gcorr = [['V15', 'V16', 'V17', 'V18', 'V21', 'V22', 'V31', 'V32', 'V33', 'V34'], ['V14'], ['V20', 'V19'], ['V24', 'V23'], ['V26', 'V25'], ['V28', 'V27'], ['V29', 'V30']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(15, 15) )
show_corr(nans_groups[4])

# %% [code]
gcorr = [['V48', 'V49'], ['V41'], ['V35', 'V36'], ['V37', 'V38'], ['V39', 'V40', 'V42', 'V43', 'V50', 'V51', 'V52'], ['V44', 'V45'], ['V46', 'V47']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(20, 20) )
show_corr(nans_groups[5])

# %% [code]
gcorr = [['V57', 'V58', 'V59', 'V60', 'V63', 'V64', 'V71', 'V72', 'V73', 'V74'], ['V65'], ['V68'], ['V53', 'V54'], ['V55', 'V56'], ['V61', 'V62'], ['V66', 'V67'], ['V69', 'V70']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(20, 20) )
show_corr(nans_groups[6])

# %% [code]
gcorr = [['V79', 'V80', 'V81', 'V84', 'V85', 'V92', 'V93', 'V94'], ['V88'], ['V89'], ['V75', 'V76'], ['V77', 'V78'], ['V82', 'V83'], ['V86', 'V87'], ['V90', 'V91']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(30,30))
show_corr(nans_groups[7])

# %% [code]
gcorr = [['V95', 'V96', 'V97', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V126', 'V127', 'V128', 'V132', 'V133', 'V134'], ['V98'], ['V107'],
         ['V100', 'V99'], ['V108', 'V109', 'V110', 'V114', 'V116'], ['V111', 'V112', 'V113'], ['V115'], ['V118', 'V117', 'V119'], ['V120', 'V122'], ['V121'],
         ['V123', 'V124', 'V125'], ['V129', 'V131'], ['V130'], ['V135', 'V136', 'V137']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
for i in nans_groups:
    print(len(i))

# %% [code]
plt.figure( figsize=(20,20))
show_corr(nans_groups[8])

# %% [code]
gcorr = [ ['V138'], ['V139', 'V140'], ['V141', 'V142'], ['V146', 'V147'], ['V148', 'V149', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158'], ['V161', 'V162', 'V163']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(15,15))
show_corr(nans_groups[9])

# %% [code]
gcorr = [['V143'], ['V144', 'V145', 'V150', 'V151', 'V152', 'V159', 'V160'], ['V164', 'V165'], ['V166']]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(25,25))
show_corr(nans_groups[10])

# %% [code]
gcorr = [
         ['V167', 'V168', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V202', 'V203', 'V204', 'V211', 'V212', 'V213'],
         ['V172', 'V186', 'V207', 'V191', 'V196', 'V174', 'V190', 'V199', 'V176', 'V193', 'V192', 'V193', 'V187'], ['V173'], ['V205', 'V206'],
         ['V214', 'V215', 'V216']
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(15,15))
show_corr(nans_groups[11])

# %% [code]
gcorr = [
         ['V169'],
         ['V170', 'V200', 'V201'], 
         ['V171'],
         ['V174'],
         ['V175'],
         ['V180'],
         ['V184', 'V185'],
         ['V188', 'V189'],
         ['V194', 'V195', 'V197', 'V198'],
         ['V208', 'V210'],
         ['V209']
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(35,35))
# plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
show_corr(nans_groups[12])

# %% [code]
gcorr = [
         ['V217', 'V218', 'V219', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237', 'V273', 'V274', 'V275'],
         ['V224', 'V225', 'V253', 'V254', 'V267', 'V228', 'V229', 'V230', 'V243', 'V246', 'V248', 'V249', 'V257', 'V258', 'V247', 'V249', 'V252'], 
         ['V226'],
         ['V223'],
         ['V240', 'V241'],
         ['V242', 'V244'],
         ['V260'],
         ['V261', 'V262'],
         ['V263', 'V264', 'V265'],
         ['V266', 'V268', 'V269'],
         ['V276', 'V278'],
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
plt.figure( figsize=(20,20))
show_corr(nans_groups[13])

# %% [code]
gcorr = [
         ['V221', 'V222', 'V227', 'V245', 'V255', 'V256', 'V259'],
         ['V238', 'V239'], 
         ['V250', 'V251'],
         ['V234'],
         ['V220'],
         ['V270', 'V271', 'V272'],
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
# 16 32 18
plt.figure( figsize=(35,35))
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
show_corr(nans_groups[14])

# %% [code]
gcorr = [
         ['V279', 'V280', 'V293', 'V294', 'V295', 'V298', 'V299', 'V306', 'V307', 'V308', 'V316', 'V317', 'V318'],
         ['V285', 'V287'], 
         ['V290', 'V291', 'V292'],
         ['V284'],
         ['V286'],
         ['V302', 'V303', 'V304'],
         ['V305'],
         ['V309', 'V311'],
         ['V310'],
         ['V312'],
         ['V319', 'V321', 'V320']
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
# 16 32 18
plt.figure( figsize=(20,20))
plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
show_corr(nans_groups[15])

# %% [code]
gcorr = [
         ['V322', 'V323', 'V324', 'V328', 'V329', 'V330', 'V331', 'V332', 'V333', 'V338', 'V337', 'V339'],
         ['V325'], 
         ['V326', 'V327'],
         ['V334', 'V334', 'V336'],
        ]

max_value_counts(gcorr, final_cols)
print(final_cols)

# %% [code]
gcorr = [['V91', 'V70', 'V30', 'V48', 'V11'], ['V127', 'V296'], ['V89', 'V68', 'V28'], ['V65', 'V88', 'V41', 'V14'], ['V80', 'V59', 'V40']]

for g in gcorr:
    final_cols = [ f for f in final_cols if f not in g]
    gval = [len(txn[str(gg)].value_counts()) for gg in g]
    most_features_index = np.argmax(gval)
    final_cols.append( g[most_features_index] )
#     print( g[most_features_index] )
final_cols

# %% [code]
gcorr = ['isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14']
plt.figure(figsize=(20, 20))
show_corr(gcorr)

# %% [code]
gcorr = [ ['C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C14'], ['C3'], ['C5', 'C9']]
final_c_cols = []

max_value_counts(gcorr, final_c_cols)

# %% [code]
final_c_cols

# %% [code]
import pprint as pp

pp.pprint(nans_dict)

# %% [code]
d = [ 'D' + str(i) for i in range(1,16)]
plt.figure( figsize = (20, 20))
show_corr(d)

# %% [code]
gcorr = [['D3', 'D5', 'D7'], ['D4', 'D6', 'D12'], ['D8'], ['D9'], ['D10'], ['D15'], ['D13'], ['D14'] ]
final_d_cols = []

max_value_counts(gcorr, final_d_cols)

# %% [code]
for l in nans_dict.values():
    if len(l) <= 3:
        print(l)

# %% [code]
vcd_cols = final_cols + final_c_cols + final_d_cols
print(vcd_cols)

# %% [code]
drop_vcd = ['V2', 'V5', 'V7', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V17', 'V18', 'V19', 'V21', 'V22', 'V23', 'V25', 'V27', 'V28', 'V29', 'V31', 'V32', 'V33', 'V34', 'V35', 'V37', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V46', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V55', 'V57', 'V58', 'V59', 'V60', 'V61', 'V63', 'V64', 'V66', 'V68', 'V69', 'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V77', 'V79', 'V81', 'V83', 'V84', 'V85', 'V87', 'V88', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105', 'V106', 'V109', 'V110', 'V112', 'V113', 'V114', 'V116', 'V117', 'V119', 'V122', 'V124', 'V125', 'V126', 'V128', 'V129', 'V132', 'V133', 'V134', 'V135', 'V137', 'V140', 'V141', 'V144', 'V145', 'V146', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V157', 'V158', 'V159', 'V161', 'V163', 'V164', 'V167', 'V168', 'V170', 'V172', 'V176', 'V177', 'V178', 'V179', 'V181', 'V182', 'V183', 'V184', 'V186', 'V187', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V199', 'V200', 'V202', 'V204', 'V206', 'V208', 'V211', 'V212', 'V213', 'V214', 'V216', 'V217', 'V218', 'V219', 'V221', 'V224', 'V225', 'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V235', 'V236', 'V237', 'V239', 'V241', 'V242', 'V243', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V252', 'V253', 'V254', 'V255', 'V256', 'V257', 'V258', 'V259', 'V262', 'V263', 'V265', 'V266', 'V269', 'V270', 'V272', 'V273', 'V275', 'V276', 'V277', 'V279', 'V280', 'V287', 'V288', 'V290', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300', 'V302', 'V304', 'V306', 'V308', 'V311', 'V313', 'V315', 'V316', 'V317', 'V318', 'V319', 'V321', 'V322', 'V323', 'V324', 'V327', 'V328', 'V329', 'V330', 'V331', 'V333', 'V334', 'V335', 'V337', 'V338', 'C1', 'C2', 'C4', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14', 'D2', 'D3', 'D4', 'D7', 'D12']

# %% [markdown]
# Preprocessing transaction data

# %% [code]
def preprocess_txn( txn ):
    # drop v, c, d columns
    
    txn = txn.drop( drop_vcd, axis = 1 )
    
    # addr1 & addr2
    
    addr1_retain_list = [299.0, 325.0, 204.0, 264.0, 330.0, 315.0, 441.0, 272.0, 123.0, 126.0, 184.0, 337.0, 191.0, 181.0, 143.0, 476.0, 310.0, 327.0, 472.0]
    txn.loc[txn.addr1.isin(addr1_retain_list) == False, 'addr1'] = "Others"
    txn.addr1.fillna('Others', inplace = True)
    txn.addr1 = txn.addr1.astype(str)
    
    addr2_retain_list = [87.0, 60.0, 96.0, 32.0, 65.0]
    txn.loc[txn.addr2.isin(addr2_retain_list) == False, 'addr2'] = "Others"
    txn.addr2.fillna('Others', inplace = True)
    txn.addr2 = txn.addr2.astype(str)
    
    # P_emaildomain
    
    txn.loc[ txn['P_emaildomain'].isin(['gmail', 'gmail.com']),'P_emaildomain'] = 'Google'
    txn.loc[ txn['P_emaildomain'].isin(['yahoo.fr', 'yahoo.es', 'yahoo.de', 'yahoo.com.mx', 'yahoo.com', 'yahoo.co.uk', 'yahoo.co.jp']),'P_emaildomain'] = 'Yahoo'
    txn.loc[ txn['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
    txn.loc[ txn['P_emaildomain'].isin(['anonymous.com']), 'P_emaildomain'] = 'Anon'
    
    p_domain_retain_list = ['protonmail.com', 'mail.com', 'aim.com', 'embarqmail.com', 'mac.com', 'cableone.net', 'icloud.com', 'bellsouth.net', 'comcast.net',
                            'suddenlink.net', 'charter.net', 'frontier.com', 'frontiernet.net', 'ymail.com', 'aol.com',
                            'Google', 'Microsoft', 'Anon', 'Yahoo']
    
    txn.loc[txn['P_emaildomain'].isin(p_domain_retain_list) == False, 'P_emaildomain'] = 'Other'
    txn.P_emaildomain.fillna('Unknown', inplace = True)
    
    # R_emaildomain
    txn.loc[ txn['R_emaildomain'].isin(['gmail', 'gmail.com']),'R_emaildomain'] = 'Google'
    txn.loc[ txn['R_emaildomain'].isin(['yahoo.fr', 'yahoo.es', 'yahoo.de', 'yahoo.com.mx', 'yahoo.com', 'yahoo.co.uk', 'yahoo.co.jp']),'R_emaildomain'] = 'Yahoo'
    txn.loc[ txn['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
    
    r_domain_retain_list = ['protonmail.com', 'mail.com', 'netzero.net', 'icloud.com', 'Google', 'Microsoft', 'suddenlink.net', 'charter.net', 'ymail.com', 'rocketmail.com', 'aim.com', 'Yahoo', 'earthlink.net']
    
    txn.loc[txn['R_emaildomain'].isin(r_domain_retain_list) == False, 'R_emaildomain'] = 'Other'
    txn.R_emaildomain.fillna('Unknown', inplace=True)
    
    # card1-6
    card = ["card" + str(i) for i in range(1, 7)]
    card_encode = { 'american express' : 3, 'discover' : 4, 'mastercard' : 2, 'visa' : 1, 'credit' : 0, 'debit' : 1, 'charge card' : 3, 'debit or credit' : 4}
    txn[card] = txn[card].replace(card_encode)
    
    # m1-9
    m = ['M' + str(i) for i in range(1,10)]
    m_encode = {'T' : 1, 'F' : 0, 'M0' : 0 , 'M1' : 1, 'M2' : 2}
    txn[m] = txn[m].replace(m_encode)
    
    
    #TransactionDT
    txn['TransactionDT'] = txn['TransactionDT']//86400

    return txn

# %% [markdown]
# Below are some helper functions which were later used in preprocessing ID data

# %% [code]
import re

#id_31
#Simplifying browser version data
def browser_version(lst):
    
    r = re.compile('[0-9][0-9]|[0-9]\.[0-9]')
    final = []
    
    browser = ""
    version = ""
    browser_list = ['firefox', 'chrome', 'ie', 'edge', 'safari', 'samsung browser', 'facebook', 'opera']
    
    for i in lst:
        
        if pd.isnull(i):
            final.append('misc')
        else:
            for br in browser_list:
                if br in i.lower():
                    browser = br
                    break
            ver = r.search(i)
            if ver:
                version = ver.group()
            else:
                version = 'generic'
    
            final.append( str(browser) )
    
#     valid = pd.DataFrame( [final, lst], index = ['new', 'old']).T
#     display(final)
    return final

#device info was a bit messed up, so we cleaned it and as follows:
def device_info(dinfo):
    brand_count = []

    for i in dinfo:
        if str(i) == 'nan':
            brand_count.append('other')
        elif 'hisense' in i.lower():
            brand_count.append('hisense')
        elif 'sm' in i.lower() or 'samsung' in i.lower() or i.lower()[:2] in ['gt', 'sc', 'sg', 'sg', 'sp']:
            brand_count.append('samsung')
        elif 'moto' in i.lower() or 'xt' in i.lower():
            brand_count.append('moto')
        elif 'lg' in i.lower() or i.lower()[:2] in ['vs']:
            brand_count.append('lg')
        elif 'trident' in i.lower():
            brand_count.append('trident')
        elif 'huawei' in i.lower() or i.lower()[:2] in ['hi']:
            brand_count.append('huawei')
        elif 'lenovo' in i.lower():
            brand_count.append('lenovo')
        elif 'rv' in i.lower():
            brand_count.append('rv')
        elif 'mac' in i.lower() or 'ios' in i.lower():
            brand_count.append('apple')
        elif 'htc' in i.lower():
            brand_count.append('htc')
        elif 'asus' in i.lower():
            brand_count.append('asus')
        elif 'windows' in i.lower() or 'rv:11.0' in i.lower():
            brand_count.append('windows')
        elif 'blade' in i.lower():
            brand_count.append('blade')
        elif 'mi' in i.lower():
            brand_count.append('xiaomi')
        elif 'linux' in i.lower():
            brand_count.append('linux')
        elif 'sony' in i.lower() or i.lower()[:2] in ['d5', 'd6', 'e2', 'e5', 'e6', 'f3', 'f5', 'f8', 'g3', 'g6', 'g8', 'h3', 'h5']:
            brand_count.append('sony')
        elif 'pixel' in i.lower() or 'nexus' in i.lower():
            brand_count.append('google')
        elif i.lower()[:2] in ['50', '60', '70', '80', '90'] or 'alcatel' in i.lower() or i.lower()[:4] in ['one ']:
            brand_count.append('alcatel')
        elif 'ilium' in i.lower():
            brand_count.append('ilium')
        elif i.lower()[:2] in ['kf']:
            brand_count.append('kindle')
        elif i.lower()[:2] in ['ta']:
            brand_count.append('nokia')
        elif i.lower()[:2] in ['z4', 'z5', 'z7', 'z8', 'z9'] or 'zte' in i.lower():
            brand_count.append('zte')
        else:
            brand_count.append('other')
            
    return brand_count

#id_33
#did a bit of FE by replacing resolutions with their categorical counterparts, this helps with encoding data as well as reduces mem usage
def resolutions( res ):
    
    def bins():
        resolutions = {
            'NA': -10,
            'SD' : 0,
            'HD' : 1280*720,
            'FHD' : 1920*1080,
            'QHD' : 2560*1440,
            '4K' : 3840*2160,
            '5K' : 5120*2880,
        }

        bins = list(resolutions.values())
        bins += [150000000]
        labels = list(resolutions.keys())
        
        return bins, labels
    
    def parse_res( res ):
        
        if pd.isnull(res):
            return -1
        mul = res.find('x')
        h = res[0:mul]
        w = res[mul+1:]
        
        return ( int(w) * int(h) )

    res = [parse_res(i) for i in res]

    bins, labels = bins()
    res = pd.cut( res, bins = bins, labels = labels)
    res = res.astype('object')
    return res

#id13
def bin13(id13):
    
    id13 = id13.fillna(-1)
    
    bins = [-10, 0, 25, 50, 100]
    labels = [0, 1, 2, 3]
    
    id13 = pd.cut( id13, bins = bins, labels = labels )
    id13 = id13.astype('float64')
    return id13

#id34
def parse34(s):
    if not pd.isnull(s):
        return s[13:]
    return s

#Preprocessing ID data: dropped some columns based on null values(>90%) while some others had jargon/unique values which didn't make sense
def preprocess_id(dd):
    
    # >90% Null Values
    # drop_list = [ 'id_18', 'id_21', 'id_22', 'id_23', 'id_26', 'id_27', 'id_07', 'id_08', 'id_25', 'id_24' ]
    drop_list = [ 'id_18', 'id_21', 'id_22', 'id_23', 'id_26', 'id_27', 'id_07', 'id_08', 'id_25', 'id_24', 'id_30',
                 'id_32', 'id_03', 'id_04', 'id_09', "id_10" ]
    dd = dd.drop(drop_list, axis = 1)
    
    #Device Type
    lol = dd[['id_31', 'DeviceType']]
    for index, row in lol.iterrows():
        if not pd.isnull(row.id_31) and pd.isnull(row.DeviceType):
            if row.id_31 == 'ie 11.0 for desktop':
                row['DeviceType'] = 'desktop'
    
    #id_31
    lol.id_31 = list( browser_version( list(dd.id_31) ) )
    dd[['id_31', 'DeviceType']] = lol[['id_31', 'DeviceType']]
    dd.DeviceType = dd.DeviceType.fillna('desktop')
    
    #Device Info
    brand_count = device_info( list(dd.DeviceInfo) )
    dd.DeviceInfo = brand_count
    
    #categorical
    #id_14
    dd.id_14 = dd.id_14/60
    
    #categorical
    #id34
    dd.id_34 = dd.id_34.apply( lambda x : parse34(x) )
    
    #id_33
    res = dd.id_33
    res = resolutions(res)
    dd.id_33 = res
    
    #id_13
    id_13 = dd.id_13
    id_13 = bin13(id_13)
    dd.id_13 = id_13
    
    #id_16
    dd.id_16 = dd.id_16.fillna('NotFound')
        
    #id_15
    dd.id_15 = dd.id_15.fillna('Unknown')
    
    #id_35,36,37,38
    dd.id_35 = dd.id_35.fillna('F')
    dd.id_36 = dd.id_36.fillna('F')
    dd.id_37 = dd.id_37.fillna('T')
    dd.id_38 = dd.id_38.fillna('F')
    
    #id_28, 29
    dd.id_28 = dd.id_28.fillna('New')
    dd.id_29 = dd.id_29.fillna('NotFound')
    
    #id_02
    id2 = dd.id_02
    id2 = id2.fillna(id2.mean())
    dd.id_02 = id2
    
    #id_05, id_06
    dd.id_05 = dd.id_05.fillna(0)
    dd.id_06 = dd.id_06.fillna(0)
    
    #id_11
    dd.id_11 = dd.id_11.fillna(100.0)
    
    #id_17, id_19, id_20
    dd.id_17 = dd.id_17.fillna( dd.id_17.mean() )
    dd.id_19 = dd.id_19.fillna( dd.id_19.mean() )
    dd.id_20 = dd.id_20.fillna( dd.id_20.mean() )
    
    return dd

# %% [code]
#Small helper function to create a submission file. Helped us save time
def submit(y_pred, num):
    sample = pd.read_csv('../input/ieee-fraud/sample_submission.csv')
    sample.isFraud = y_pred
    sub_name = 'submission_' + str(num) + '.csv'
    sample.to_csv(sub_name, index=False)

#Used to weight the number of positives with the number of negatives, like a weighted loss function
def get_scale_pos_weight(y):
    return (y.value_counts().iloc[0])/(y.value_counts().iloc[1])

# %% [markdown]
# FINALLY: Using all the functions above, reading all the data and processing it in the function below

# %% [code]
def proc():
    tt = pd.read_csv('../input/ieee-fraud/train_transaction.csv', index_col='TransactionID')
    ti = pd.read_csv('../input/ieee-fraud/train_identity.csv', index_col='TransactionID')
    
    tt = preprocess_txn(tt)
    ti = preprocess_id(ti)
    
    t1 = pd.merge(tt, ti, left_index=True, right_index=True, how='left')
    obj = t1.dtypes[ t1.dtypes == 'object' ].index.tolist()
    t1[obj] = t1[obj].fillna('nullval')
    
    x = t1.sort_values('TransactionDT').drop('isFraud', axis = 1)
    y = pd.DataFrame( t1.sort_values('TransactionDT').isFraud )
    
    del tt, ti, t1
    
    gc.collect()
    
    #Creating a list of categorical columns(having dtype object/category)
    obj_list = x.dtypes[x.dtypes=='object'].index.tolist()
    obj_list += x.dtypes[x.dtypes=='category'].index.tolist()

    #Using ColumnTransformer and one hot encoding to encode the categorical columns
    columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), obj_list)], remainder='passthrough')
    
    x = x.fillna(-999)

    x = np.array(columnTransformer.fit_transform(x))
    
    #Reading test data and doing the same preprocessing on it
    test_txn = pd.read_csv('../input/ieee-fraud/test_transaction.csv', index_col = 'TransactionID')
    test_id = pd.read_csv('../input/ieee-fraud/test_identity.csv', index_col = 'TransactionID')
    
    test_txn = preprocess_txn(test_txn)
    test_id = preprocess_id(test_id)
    
    test = pd.merge(test_txn, test_id, left_index=True, right_index=True, how='left')
    
    test[obj] = test[obj].fillna('nullval')
    
    test = test.fillna(-999)
    
    test = pd.DataFrame( np.array(columnTransformer.transform(test)) )

    features = columnTransformer.get_feature_names()
    
    x = pd.DataFrame(x, columns = features)
    test.columns = features
    
    display( x.head() )
    display( y.head() )
    display( test.head() )

    #Splitting our train data into 75% train and 25% validation dataset
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.25, random_state=42, stratify=y)
    
    data = {
        'x_train' : x_train,
        'y_train' : y_train,
        'x_val' : x_val,
        'y_val' : y_val,
        'test' : test,
        'x' : x,
        'y' : y
    }
    return data

# %% [code]
data = proc()

# %% [markdown]
# ## HyperOpt Begins

# %% [markdown]
# Used Hyperopt to get a general idea of how the parameters are to be tuned. Tuned a few hyperparameters manually based on evaluation metrics on the validation set as we see later on

# %% [code]
def optimize(params, x_train, y_train, x_test, y_test):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        missing = -999,
        eval_metric='auc',
        random_state = 42,
        scale_pos_weight = get_scale_pos_weight(data['y_train']),
        verbosity=1,
        tree_method='gpu_hist',
        **params)
    
    
    model.fit(x_train, y_train)
    y_pred = model.predict_proba(x_test)
    score = roc_auc_score(y_test, y_pred[:, 1])
    
    del x_train, y_train, x_test, y_test, y_pred
    gc.collect()
    
    return -(score)


param_space = {
    'max_depth' : scope.int(hp.quniform("max_depth", 4, 16, 2)),
    'n_estimators' : scope.int(hp.quniform("n_estimators", 500, 1000, 1)),
    'learning_rate' : hp.uniform("learning_rate", 0.01, 0.2),
    'alpha' : hp.quniform('alpha', 5, 16, 1),
    'subsample' : hp.quniform('subsample',0,1, 0.01)
}

trial = Trials()

optimization_function = partial( optimize,x_train =data['x_train'],y_train=data['y_train'].isFraud.ravel(),x_test = data['x_val'],y_test=data['y_val'].isFraud.ravel())


# %% [code]
result = fmin(
    fn=optimization_function,
    space=param_space,
    trials=trial,
    algo=tpe.suggest,
    max_evals=70
)

# %% [markdown]
# ## HyperOpt Ends

# %% [markdown]
# # FINAL MODEL
# Now our final model was created based on 2 different models which was then averaged and then power averaged.
# We used an lgbm model with a few FE added by a group member, and an xgboost model based on normal preprocessing defined above for just the normal average.
# For power averaging, we used 3 models: 2 mentioned above, and a third xgboost model having different params which we will be defining below

# %% [code]
result

# %% [markdown]
# Now running our model on train set(not the whole dataset) and checking how our validation loss looks like on our manually chosen hyperparameters(informed decision from hyperopt)

# %% [code]

params = {
    'alpha' : 7,
    'colsample_bytree' : 0.8,
    'colsample_bynode' : 0.8,
    'colsample_bylevel' : 0.8,
    'learning_rate' : 0.01,
    'max_depth' : 16,
    'n_estimators' : 4000,
#     'gamma' : 5,
    'subsample' : 0.8,
    'predictor' : 'gpu_predictor'
}

model = xgb.XGBClassifier(
        objective='binary:logistic',
        missing = -999,
        eval_metric=["error", "auc", "logloss"],
        random_state = 42,
        scale_pos_weight = get_scale_pos_weight(data['y_train']),
        tree_method = 'gpu_hist',
        gpu_id = 0,
        verbosity=1,
        **params)

eval_set = [(data['x_train'], data['y_train'].isFraud.ravel()), (data['x_val'], data['y_val'].isFraud.ravel())]

model.fit(data['x_train'], data['y_train'].isFraud.ravel(), eval_set=eval_set, verbose=True)

# %% [code]
results = model.evals_result()

# %% [code]
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC Score')
plt.title('XGBoost AUC Score')
plt.show()

# # plot classification error
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Logloss')
plt.title('XGBoost Logloss')
plt.show()

# # plot classification error
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# %% [code]
model

# %% [code]
epochs = len(results['validation_0']['auc'])
x_axis = range(0, epochs)

# plot log loss
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['auc'], label='Train')
ax.plot(x_axis, results['validation_1']['auc'], label='Test')
ax.legend()
plt.ylabel('AUC Score')
plt.title('XGBoost AUC Score')
plt.show()

# # plot classification error
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Logloss')
plt.title('XGBoost Logloss')
plt.show()

# # plot classification error
fig, ax = plt.subplots(figsize=(20, 20))
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')
plt.show()

# %% [markdown]
# Checking ROC AUC Score:

# %% [code]
y_pred_val = model.predict_proba(data['x_val'])[:, 1]
y_pred_train = model.predict_proba(data['x_train'])[:, 1]

print(f"ROC AUC Score\nTrain : {round(roc_auc_score( data['y_train'], y_pred_train ), 7)}\nTest : {round( roc_auc_score( data['y_val'], y_pred_val ), 7 )}")

# %% [markdown]
# Now running the model on the whole dataset and predicting on test data

# %% [code]
import xgboost as xgb

params = {
    'alpha' : 7,
    'colsample_bytree' : 0.8,
    'colsample_bynode' : 0.8,
    'colsample_bylevel' : 0.8,
    'learning_rate' : 0.01,
    'max_depth' : 16,
    'n_estimators' : 3000,
#     'gamma' : 5,
    'subsample' : 0.8,
    'predictor' : 'gpu_predictor'
}

model = xgb.XGBClassifier(
        objective='binary:logistic',
        missing = -999,
        eval_metric='auc',
        random_state = 42,
        scale_pos_weight = get_scale_pos_weight(data['y']),
        tree_method = 'gpu_hist',
        gpu_id = 0,
        verbosity=1,
        **params)

model.fit(data['x'], data['y'].isFraud.ravel())

# %% [code]
model

# %% [code]
y_pred_train = model.predict_proba(data['x'])[:, 1]
y_pred_train

# %% [code]
print(f"ROC AUC Score\nTrain : {round(roc_auc_score( data['y'], y_pred_train ), 7)}")

# %% [code]
y_pred_test1 = model.predict_proba(data['test'])[:, 1]
y_pred_test1

# %% [code]
submit(y_pred_test1, 36)

# %% [markdown]
# We submitted this model as an independent submission and got a score of 0.96323 on public leaderboard. We tried many variations of this model by tuning hyperparameters but the score did not improve much. So we used 2 variations of this model, decided to use aggregation on our skewed data and used LGBM with the hopes that our scores would improve

# %% [markdown]
# Below is the variation 2 of our xgboost model: This model got us a score of 0.96382 on public leaderboard

# %% [code]
import xgboost as xgb

params2 = {
    'alpha' : 7,
#     'colsample_bytree' : 0.8,
    'learning_rate' : 0.01,
    'max_depth' : 16,
    'n_estimators' : 3000,
#     'gamma' : 5,
    'subsample' : 0.8
}
model = xgb.XGBClassifier(
        objective='binary:logistic',
        missing = -999,
        eval_metric='auc',
        random_state = 42,
        scale_pos_weight = get_scale_pos_weight(data['y']),
        tree_method = 'gpu_hist',
        gpu_id = 0,
        verbosity=1,
        **params2)

model.fit(data['x'], data['y'].isFraud.ravel())

# %% [code]
y_pred_test2 = model.predict_proba(data['test'])[:, 1]
y_pred_test2

# %% [code]
submit(y_pred_test2, 37)

# %% [markdown]
# LGBM Model Preprocessing+Model
# One of our group members did a bit different preprocessing and added a few features like doing aggregation to remove skewness of data and make the model more generalized.

# This model was also used in our final submission. Not making any changes to the code by the member to maintain sanctity and sequence to the script


# %% [code]

##referenced from stackoverflow as it is
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
##referenced from stackoverflow as it is: END

def loading_data():
    print("load data")
    train_txn = pd.read_csv("/kaggle/input/ieee-fraud/train_transaction.csv")
    train_id = pd.read_csv("/kaggle/input/ieee-fraud/train_identity.csv")

    test_txn = pd.read_csv("/kaggle/input/ieee-fraud/test_transaction.csv")
    test_id = pd.read_csv("/kaggle/input/ieee-fraud/test_identity.csv")

    fix_col_name = {testIdCol:trainIdCol for testIdCol, trainIdCol in zip(test_id.columns, train_id.columns)}
    test_id.rename(columns=fix_col_name, inplace=True)
    
    train_txn = reduce_mem_usage(train_txn)
    train_id = reduce_mem_usage(train_id)

    train = train_txn.merge(train_id, on='TransactionID', how='left')
    test = test_txn.merge(test_id, on='TransactionID', how='left')

    tt = pd.concat([train, test], ignore_index=True)
    gc.collect()
    del train_txn, train_id, test_txn, test_id; x = gc.collect()  
    return tt

# %% [markdown]
# Some helper functions, same as the ones we created earlier

# %% [code]
def parse34(s):
    if not pd.isnull(s):
        return s[13:]
    return s

def resolutions( res ):
    
    def bins():
        resolutions = {
            'NA': -10,
            'SD' : 0,
            'HD' : 1280*720,
            'FHD' : 1920*1080,
            'QHD' : 2560*1440,
            '4K' : 3840*2160,
            '5K' : 5120*2880,
        }

        bins = list(resolutions.values())
        bins += [150000000]
        labels = list(resolutions.keys())
        
        return bins, labels
    
    def parse_res( res ):
        
        if pd.isnull(res):
            return -1
        mul = res.find('x')
        h = res[0:mul]
        w = res[mul+1:]
        
        return ( int(w) * int(h) )

    res = [parse_res(i) for i in res]

    bins, labels = bins()
    res = pd.cut( res, bins = bins, labels = labels)
    res = res.astype('object')
    return res

def bin13(id13):
    
    id13 = id13.fillna(-1)
    
    bins = [-10, 0, 25, 50, 100]
    labels = [0, 1, 2, 3]
    
    id13 = pd.cut( id13, bins = bins, labels = labels )
    id13 = id13.astype('float64')
    return id13

# %% [markdown]
# Preprocessing data: Most of the preprocessing is same as the proc() function above. Added aggregation to a few columns to help with model generalization

# %% [code]
def processing_data(tt):
    drop_col_list = []
    print("process data: start")

    tt.id_14 = tt.id_14/60
    
    #categorical
    #id34
    tt.id_34 = tt.id_34.apply( lambda x : parse34(x) )
    
    #id_33
    res = tt.id_33
    res = resolutions(res)
    tt.id_33 = res
    
    #id_13
    id_13 = tt.id_13
    id_13 = bin13(id_13)
    tt.id_13 = id_13
    
    #id_16
    tt.id_16 = tt.id_16.fillna('NotFound')
    
    #id_15
    tt.id_15 = tt.id_15.fillna('Unknown')
    
    #id_35,36,37,38
    tt.id_35 = tt.id_35.fillna('F')
    tt.id_36 = tt.id_36.fillna('F')
    tt.id_37 = tt.id_37.fillna('T')
    tt.id_38 = tt.id_38.fillna('F')
    
    #id_28, 29
    tt.id_28 = tt.id_28.fillna('New')
    tt.id_29 = tt.id_29.fillna('NotFound')
    
    #id_02
    id2 = tt.id_02
    id2 = id2.fillna(id2.mean())
    tt.id_02 = id2
    gc.collect()
    #id_05, id_06
    tt.id_05 = tt.id_05.fillna(0)
    tt.id_06 = tt.id_06.fillna(0)
    
    #id_11
    tt.id_11 = tt.id_11.fillna(100.0)
    
    #id_17, id_19, id_20
    tt.id_17 = tt.id_17.fillna( tt.id_17.mean() )
    tt.id_19 = tt.id_19.fillna( tt.id_19.mean() )
    tt.id_20 = tt.id_20.fillna( tt.id_20.mean() )
    
    # TransactionDT
    START_DATE = '2015-04-22'
    startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
    tt['NewDate'] = tt['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds=x)))
    tt['NewDate_Year_Month_Date'] = tt['NewDate'].dt.year.astype(str) + '-' + tt['NewDate'].dt.month.astype(str) + '-' + tt['NewDate'].dt.day.astype(str)
    tt['NewDate_Year_Month'] = tt['NewDate'].dt.year.astype(str) + '-' + tt['NewDate'].dt.month.astype(str)
    tt['NewDate_Week_day'] = tt['NewDate'].dt.dayofweek
    tt['NewDate_Hour'] = tt['NewDate'].dt.hour
    tt['NewDate_Day'] = tt['NewDate'].dt.day
    drop_col_list.extend(["TransactionDT","NewDate"])
    gc.collect()
    # TransactionAMT
    tt['Amt'] = (tt['TransactionAmt'] - np.floor(tt['TransactionAmt'])).astype('float32')
    tt['Amt_Bin'] = pd.qcut(tt['TransactionAmt'],15)
    #separating 'TransactionAmt' into 15 quantile bins
    
    #cardX
    card_cols = [c for c in tt if c[0:2] == 'ca']
    for col in ['card2','card3','card4','card5','card6']:
        tt[col] = tt.groupby(['card1'])[col].transform(lambda x: x.mode(dropna=False).iat[0])
        tt[col].fillna(tt[col].mode()[0], inplace=True)

    tt['DeviceInfo'] = tt['DeviceInfo'].fillna('unknown_device').str.lower()
    tt['DeviceInfo'] = tt['DeviceInfo'].str.split('/', expand=True)[0]
    tt.loc[tt['DeviceInfo'].str.contains('rv:', na=False), 'DeviceInfo'] = 'RV'
    tt.loc[tt['DeviceInfo'].str.contains('HUAWEI'or 'hi' or 'ALE-' or '-L', na=False), 'DeviceInfo'] = 'Huawei'
    tt.loc[tt['DeviceInfo'].str.contains('Blade' or 'BLADE', na=False), 'DeviceInfo'] = 'ZTE'
    tt.loc[tt['DeviceInfo'].str.contains('Moto G' or 'Moto' or 'moto', na=False), 'DeviceInfo'] = 'Motorola'
    tt.loc[tt['DeviceInfo'].str.contains('SAMSUNG'or 'GT' or 'SC'or 'sg'or 'SG'or'SP' or 'SM' or 'GT-', na=False), 'DeviceInfo'] = 'Samsung'
    tt.loc[tt['DeviceInfo'].str.contains('HTC' or "0paj5"or "0pja2"or "0pm92"or "2pq93"or "2ps64"or "2pyb2" or "2pzc5", na=False), 'DeviceInfo'] = 'HTC'
    tt.loc[tt['DeviceInfo'].str.contains('ASUS', na=False), 'DeviceInfo'] = 'Asus'
    tt.loc[tt['DeviceInfo'].str.contains('LG-'or "lg"or "nexus"or "lm-"or "vs"or "VS" or "LM-" or 'Nexus' or 'NEXUS', na=False), 'DeviceInfo'] = 'LG'
    tt.loc[tt['DeviceInfo'].str.contains("kfa"or "KFD"or "KFF" or "KFG"or "KFJ"or "KFK"or "KFM"or "KFS"or "KFT", na=False), 'DeviceInfo'] = 'Amazon'
    tt.loc[tt['DeviceInfo'].str.contains('Linux', na=False), 'DeviceInfo'] = 'Linux'
    tt.loc[tt['DeviceInfo'].str.contains('XT', na=False), 'DeviceInfo'] = 'Sony'
    tt.loc[tt['DeviceInfo'].isin(tt['DeviceInfo'].value_counts()[tt['DeviceInfo'].value_counts() < 1000].index), 'DeviceInfo'] = "Others"

    # V1 - V339
    v_cols = [c for c in tt if c[0] == 'V']
    v_nan_df = tt[v_cols].isna()
    nan_groups={}

    #doing a PCA to see which are our most imp features
    for col in v_cols:
        cur_group = v_nan_df[col].sum()
        try:
            nan_groups[cur_group].append(col)
        except:
            nan_groups[cur_group]=[col]
    del v_nan_df; x=gc.collect()

    for nan_cnt, v_group in nan_groups.items():
        tt['V'+str(nan_cnt)+'nan'] = nan_cnt
        sc = preprocessing.MinMaxScaler()
        pca = PCA(n_components=2)
        v_group_pca = pca.fit_transform(sc.fit_transform(tt[v_group].fillna(-1)))
        tt['V'+str(nan_cnt)+'pca'] = v_group_pca[:,0]
        tt['V'+str(nan_cnt)+'_pca'] = v_group_pca[:,1]
        
     # P_emaildomain
    gc.collect()
    tt.loc[ tt['P_emaildomain'].isin(['gmail', 'gmail.com']),'P_emaildomain'] = 'Google'
    tt.loc[ tt['P_emaildomain'].isin(['yahoo.fr', 'yahoo.es', 'yahoo.de', 'yahoo.com.mx', 'yahoo.com', 'yahoo.co.uk', 'yahoo.co.jp']),'P_emaildomain'] = 'Yahoo'
    tt.loc[ tt['P_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'P_emaildomain'] = 'Microsoft'
    tt.loc[ tt['P_emaildomain'].isin(['anonymous.com']), 'P_emaildomain'] = 'Anon'

    p_domain_retain_list = ['protonmail.com', 'mail.com', 'aim.com', 'embarqmail.com', 'mac.com', 'cableone.net', 'icloud.com', 'bellsouth.net', 'comcast.net',
                            'suddenlink.net', 'charter.net', 'frontier.com', 'frontiernet.net', 'ymail.com', 'aol.com',
                            'Google', 'Microsoft', 'Anon', 'Yahoo']
    
    tt.loc[tt['P_emaildomain'].isin(p_domain_retain_list) == False, 'P_emaildomain'] = 'Other'
    tt.P_emaildomain.fillna('Unknown', inplace = True)
    
    # R_emaildomain
    tt.loc[ tt['R_emaildomain'].isin(['gmail', 'gmail.com']),'R_emaildomain'] = 'Google'
    tt.loc[ tt['R_emaildomain'].isin(['yahoo.fr', 'yahoo.es', 'yahoo.de', 'yahoo.com.mx', 'yahoo.com', 'yahoo.co.uk', 'yahoo.co.jp']),'R_emaildomain'] = 'Yahoo'
    tt.loc[ tt['R_emaildomain'].isin(['hotmail.com','outlook.com','msn.com', 'live.com.mx', 'hotmail.es','hotmail.co.uk', 'hotmail.de','outlook.es', 'live.com', 'live.fr','hotmail.fr']), 'R_emaildomain'] = 'Microsoft'
    
    r_domain_retain_list = ['protonmail.com', 'mail.com', 'netzero.net', 'icloud.com', 'Google', 'Microsoft', 'suddenlink.net', 'charter.net', 'ymail.com', 'rocketmail.com', 'aim.com', 'Yahoo', 'earthlink.net']
    
    tt.loc[tt['R_emaildomain'].isin(r_domain_retain_list) == False, 'R_emaildomain'] = 'Other'
    tt.R_emaildomain.fillna('Unknown', inplace=True)
    

    drop_col_list.extend(v_cols)
    gc.collect()

    tt['card1_card2']=tt['card1'].astype(str)+'_'+tt['card2'].astype(str)
    tt['card3_card4']=tt['card3'].astype(str)+'_'+tt['card4'].astype(str)
    tt['card5_card6']=tt['card5'].astype(str)+'_'+tt['card6'].astype(str)

    tt['addr1_addr2']=tt['addr1'].astype(str)+'_'+tt['addr2'].astype(str)
    tt['New_card1_card2_addr1_addr2']=tt['card1'].astype(str)+'_'+tt['card2'].astype(str)+'_'+tt['addr1'].astype(str)+'_'+tt['addr2'].astype(str)

    tt['New_P_emaildomain_addr1'] = tt['P_emaildomain'] + '_' + tt['addr1'].astype(str)
    tt['New_R_emaildomain_addr2'] = tt['R_emaildomain'] + '_' + tt['addr2'].astype(str)
    
    #Feature Eng

    
    ## Basic idea of below aggregation: Every card has some variable transaction amount attached to it. 
    #It's better to study this Amount separately for each card rather than to read the two individually. 
    #This reduces the number of features we would have to study (dimension reduction), 
    #Removes any extra data points in either which might not be useful, 
    #also helps in getting rid of the skewness that the data brings along with it.
    gc.collect()
    tt.plot.scatter(x='card1', y='TransactionAmt', s = 2,colormap='viridis') 
    tt.plot.scatter(x='card2', y='TransactionAmt', s = 2,colormap='viridis')
    tt.plot.scatter(x='card3', y='TransactionAmt', s = 2,colormap='viridis')
    tt.plot.scatter(x='card4', y='TransactionAmt', s = 2,colormap='viridis')
    gc.collect()
    
    #Depicts Skewed Probability Density of 'card1'
    
#     plt.figure(figsize=(12,6))
#     sns.distplot(tt['card1'].apply(np.log))
#     plt.title("Log Transformation of TransactionAmt Distribution ")
#     plt.ylabel("Probability Density")
#     plt.show()
    tt['TransactionAmt_mean_card1'] = tt['TransactionAmt'] / tt.groupby(['card1'])['TransactionAmt'].transform('mean')
    tt['TransactionAmt_mean_card4'] = tt['TransactionAmt'] / tt.groupby(['card4'])['TransactionAmt'].transform('mean')
    tt['TransactionAmt_mean_card2'] = tt['TransactionAmt'] / tt.groupby(['card2'])['TransactionAmt'].transform('mean')
    tt['TransactionAmt_mean_card3'] = tt['TransactionAmt'] / tt.groupby(['card3'])['TransactionAmt'].transform('mean')
    tt['TransactionAmt_std_card1'] = tt['TransactionAmt'] / tt.groupby(['card1'])['TransactionAmt'].transform('std')
    tt['TransactionAmt_std_card4'] = tt['TransactionAmt'] / tt.groupby(['card4'])['TransactionAmt'].transform('std')
    tt['TransactionAmt_std_card2'] = tt['TransactionAmt'] / tt.groupby(['card2'])['TransactionAmt'].transform('std')
    tt['TransactionAmt_std_card3'] = tt['TransactionAmt'] / tt.groupby(['card3'])['TransactionAmt'].transform('std')
    gc.collect()
    #after aggregation of the above, we have normalised the columns
    
    
#     plt.figure(figsize=(12,6))
#     sns.distplot(tt['TransactionAmt_mean_card1'].apply(np.log))
#     plt.title("Log Transformation of TransactionAmt Distribution ")
#     plt.ylabel("Probability Density")
#     plt.show()
    
    
#     tt.plot.scatter(x='card1', y='id_02' ,c = 'isFraud' ,s = 2,colormap='viridis')
    
    #Depicts Skewed Nature of 'id_02'
#     plt.figure(figsize=(12,6))
#     sns.distplot(tt['id_02'].apply(np.log))
#     plt.title("Log Transformation of TransactionAmt Distribution ")
#     plt.ylabel("Probability Density")
#     plt.show()
    #########
    tt = reduce_mem_usage(tt)
    gc.collect()
    #id_02 contains the most information in the identity train dataset, every id_02 cell can be studied alonside
    #a card/transaction to see if there is a corr() there.
    
# isFraud
# 0    0.000508
# 1    0.044691
# Name: (id_02, card1), dtype: float64

# there is a good correlation there so we can go ahead with aggregating these
    tt['id_02_mean_card1'] = tt['id_02'] / tt.groupby(['card1'])['id_02'].transform('mean')
    tt['id_02_mean_card2'] = tt['id_02'] / tt.groupby(['card2'])['id_02'].transform('mean')
    tt['id_02_mean_card3'] = tt['id_02'] / tt.groupby(['card3'])['id_02'].transform('mean')
    tt['id_02_mean_card4'] = tt['id_02'] / tt.groupby(['card4'])['id_02'].transform('mean')
    gc.collect()
    tt['id_02_std_card1'] = tt['id_02'] / tt.groupby(['card1'])['id_02'].transform('std')
    tt['id_02_std_card2'] = tt['id_02'] / tt.groupby(['card2'])['id_02'].transform('std')
    tt['id_02_std_card3'] = tt['id_02'] / tt.groupby(['card4'])['id_02'].transform('std')
    tt['id_02_std_card4'] = tt['id_02'] / tt.groupby(['card4'])['id_02'].transform('std')
    ####After Aggregation, we can get rid of this skew in id_02
    tt['id_02_mean_card1'] = tt['id_02'] / tt.groupby(['card1'])['id_02'].transform('mean')
    plt.figure(figsize=(12,6))
    sns.distplot(tt['id_02_mean_card1'].apply(np.log))
    plt.title("Log Transformation of TransactionAmt Distribution ")
    plt.ylabel("Probability Density")
    plt.show()
    

    
    gc.collect()
    tt.plot.scatter(x='card1', y='isFraud' ,c = 'D15' ,s = 2,colormap='viridis')
    tt.plot.scatter(x='card2', y='isFraud' ,c = 'D15' ,s = 2,colormap='viridis')
    tt.plot.scatter(x='card3', y='isFraud' ,c = 'D15' ,s = 2,colormap='viridis')
    
    
    
# isFraud
# 0   -0.005782
# 1    0.041845
# Name: (D15, card1), dtype: float64

# there is a good correlation there so we can go ahead with aggregating these as well
    gc.collect()
    tt['D15_mean_card1'] = tt['D15'] / tt.groupby(['card1'])['D15'].transform('mean')
    tt['D15_mean_card2'] = tt['D15'] / tt.groupby(['card2'])['D15'].transform('mean')
    tt['D15_mean_card3'] = tt['D15'] / tt.groupby(['card3'])['D15'].transform('mean')
    tt['D15_mean_card4'] = tt['D15'] / tt.groupby(['card4'])['D15'].transform('mean')
    tt['D15_std_card2'] = tt['D15'] / tt.groupby(['card2'])['D15'].transform('std')
    tt['D15_std_card3'] = tt['D15'] / tt.groupby(['card3'])['D15'].transform('std')
    tt['D15_std_card1'] = tt['D15'] / tt.groupby(['card1'])['D15'].transform('std')
    tt['D15_std_card4'] = tt['D15'] / tt.groupby(['card4'])['D15'].transform('std')
    tt = reduce_mem_usage(tt)
#     vis4 = tt.groupby(['addr1'],['isFraud']).sum()
#     print(vis4)


# isFraud
# 0    0.017324
# 1   -0.018763
# Name: (D15, addr1), dtype: float64
        
        
# isFraud
# 0    0.054641
# 1    0.081718
# Name: (D15, addr2), dtype: float64

# Correlation between D15,addr1/addr2 is good when grouped by Fraud, we can aggregate these as well
    tt['D15_mean_addr1'] = tt['D15'] / tt.groupby(['addr1'])['D15'].transform('mean')
    tt['D15_mean_addr2'] = tt['D15'] / tt.groupby(['addr2'])['D15'].transform('mean')
    tt['D15_mean_addr1'] = tt['D15'] / tt.groupby(['addr1'])['D15'].transform('std')
    tt['D15_mean_addr2'] = tt['D15'] / tt.groupby(['addr2'])['D15'].transform('std')
    
    drop_col_list.extend(card_cols)
    gc.collect()
 
    #Drop all columns which have >85% null values
    
    dropping = ['id_24','id_25', 'id_23', 'id_07', 'id_08','id_21','id_22','id_27','id_26','dist2','D7','id_18','D13','D14','D12','D6','D8','D9','V151','V150','V153','V154','V149','V155','V156','V152','V144','V157','V159','V147','V160','V161','V162','V148','V158','V146','V163','V143','V142','V164','V141','V140','V139','V138','V145','V327','V332','V331','V165','V338','V337','V330','V329','V328','V324','V326','V325','V335','V336','V323','V322','V166','V334','V333','V339']
    drop_col_list.extend(dropping)
#         lis = ['id_13','id_16', 'V268', 'V269','V225','V224','V223','V258','V260','V278','V277', 'V276','V267','V226','V274','V273','V261','V262','V263','V264','V253','V228','V229','V241','V249','V248','V247','V246','V254','V244','V243','V242','V257','V230','V219','V240','V237','V236','V235','V233','V232','V231','V252','V266','V218','V265','V217','id_06','id_05','id_20','id_19','id_17','V186','V191','V193','V182','V183','V178','V192','V173','V190','V177','V187','V176','V196','V172','V199','V181','V213','V216','V215','V179','V167','V202','V203','V204','V205','V168','V214','V206','V207','V211','V212','V169','V175','V171','V170','V174','V184','V180','V194','V210','V209','V208','V201','V198','V197','V195','V200','V188','V189','V185','id_31','id_02','id_15','id_37','id_36','id_35','id_29','id_28','id_28','id_38','id_11','V239','V238','V251','V220','V221','V259','V222','V245','V255','V250','V227','V234','V256','V270','V271','V272','id_01','id_12']
#     lis = ['V268', 'V269','V225','V224','V223','V258','V260','V278','V277', 'V276','V267','V226','V274','V273','V261','V262','V263','V264','V253','V228','V229','V241','V249','V248','V247','V246','V254','V244','V243','V242','V257','V230','V219','V240','V237','V236','V235','V233','V232','V231','V252','V266','V218','V265','V217','V186','V191','V193','V182','V183','V178','V192','V173','V190','V177','V187','V176','V196','V172','V199','V181','V213','V216','V215','V179','V167','V202','V203','V204','V205','V168','V214','V206','V207','V211','V212','V169','V175','V171','V170','V174','V184','V180','V194','V210','V209','V208','V201','V198','V197','V195','V200','V188','V189','V185','V239','V238','V251','V220','V221','V259','V222','V245','V255','V250','V227','V234','V256','V270','V271','V272']
#     drop_col_list.extend(lis)
    # Frequency Encoding 
    fe=["Amt_Bin",'P_emaildomain','R_emaildomain','DeviceType','DeviceInfo','card4','card6']+[c for c in tt if c[0] == 'M']
    for i in fe:
        v = tt[i].value_counts(dropna=True, normalize=True).to_dict()
        v[-1] = -1
        n = i+'_FE'
        tt[n] = tt[i].map(v)
        tt[n] = tt[n].astype('float32')

    tt=tt.drop(drop_col_list, axis=1)
    gc.collect()
    for col in tt.columns:
        if tt[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(tt[col].astype(str).values))
            tt[col] = le.transform(list(tt[col].astype(str).values))
            
    tt = reduce_mem_usage(tt)
    print("proc data: end")

#     print(tt.group_by[])
    
    return tt

# %% [markdown]
# FINAL LGBM MODEL

# %% [code]
def modeling(tt,target):
    print("model: start")
    gc.collect()
    train = tt[tt[target].notnull()]
    test = tt[tt[target].isnull()]
    #used KFold validation to improve the model as much as possible
    folds = KFold(n_splits = 10, shuffle = True, random_state = 1001)

    pred1 = np.zeros(train.shape[0])
    pred_test = np.zeros(test.shape[0])
    
    feature_importance_df = pd.DataFrame()

    features = [f for f in train.columns if f not in [target,'TransactionID','Amt_Bin','NewDate']]

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[features], train[target])):
        
        start_time = time.time()
        print('fold {}'.format(n_fold + 1))

        X_train, y_train = train[features].iloc[train_idx], train[target].iloc[train_idx]

        X_valid, y_valid = train[features].iloc[valid_idx], train[target].iloc[valid_idx]
        
        params={"boosting_type": "gbdt",
    "colsample_bytree": 0.5,
    "early_stopping_rounds": 100,
    "learning_rate": 0.09,
    "max_bin": 255,
    "max_depth": -1,
    "metric": "auc",
    "n_jobs": -1,
    "num_leaves": 2**11,
    "objective": "binary",
    "seed": 1337,
    "subsample": 0.8,
    "subsample_freq": 1,
    "tree_learner": "serial" }
       
        clf = LGBMClassifier(**params, n_estimators=20000) 

        clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_valid, y_valid)], 
                eval_metric = 'auc', verbose = 200, early_stopping_rounds = 200)

        #y_pred_valid
        pred1[valid_idx] = clf.predict_proba(X_valid, num_iteration=clf.best_iteration_)[:, 1]
        pred_test += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)


        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(y_valid, pred1[valid_idx]))) 


    print('Full AUC score %.6f' % roc_auc_score(train[target], pred1))
    submit(pred_test, 38)

# %% [code]
tt = loading_data()
tt = processing_data(tt)
modeling(tt, 'isFraud')

# %% [markdown]
# # For Final Submission
# Used the predictions of the 2 xgboost models and the lgbm model to create the final submissions
# 
# 
# For power averaging(score: 0.96500 on public data) : used lgbm and the first xgboost
# 
# For normal averaging(score: 0.96468 on public data) : used lgbm and both xgboost

# %% [code]
sub36=pd.read_csv("./submission_36.csv")
sub37 = pd.read_csv("./submission_37.csv")
sub38=pd.read_csv("./submission_38.csv")

# %% [markdown]
# Power averaging:

# %% [code]
sub36['isFraud'] = ((sub36['isFraud']**2)+(sub37['isFraud']**2)+(sub38["isFraud"]**2))/3

# %% [code]
sub36.to_csv('submission_powavg.csv', index=False)

# %% [markdown]
# Simple Averaging:

# %% [code]
sub36=pd.read_csv("./submission_36.csv")
sub36['isFraud'] = (sub36['isFraud']+sub38['isFraud'])/2

# %% [code]
sub36.to_csv('submission_avg.csv', index=False)