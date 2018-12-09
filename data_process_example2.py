import pandas as pd
import numpy as np
from functools import reduce
import rare_cleaner
import os
import geohash
from sklearn.preprocessing import LabelEncoder
import utils


nrows = None
ssh = True

if ssh:
    file_path = '/data1/jianglibin/orange/data/'
else:
    file_path = 'data'

train_tr = pd.read_csv(os.path.join(file_path,'transaction_train_new.csv'),nrows = nrows)
test_tr = pd.read_csv(os.path.join(file_path,'transaction_round1_new.csv'),nrows = nrows)
tag = pd.read_csv(os.path.join(file_path,'tag_train_new.csv'))

print('append extra cols...')
train_tr = utils.append_cols(train_tr,mode='tran')
test_tr = utils.append_cols(test_tr,mode='tran')

train_num = train_tr.shape[0]
test_num = test_tr.shape[0]
data_df = pd.concat([train_tr,test_tr],axis=0).reset_index(drop=True)
drop_cols = ['code1','code2','acc_id1','acc_id2','acc_id3','device_code1','device_code2','device_code3','ip1']#,'market_code','market_type']
# drop_cols = [''' ''code1','code2',''' 'acc_id1','acc_id2','acc_id3','device_code1','device_code2','device_code3','ip1','market_code',] + ['device1','mac1','ip1_sub']
data_df.drop(columns=drop_cols,inplace=True)

print('keep both value...')
data_df = utils.save_both_value(data_df,utils.tran_variable.tran_both_value)

print('process decive2...')
data_df['device2'] = data_df['device2'].apply(lambda x: utils.convert_device2(x) if pd.notnull(x) else x)

print('process geo_code...')
data_df['geo_code'].fillna('0000',inplace=True)
data_df['geo_x'] = data_df['geo_code'].apply(lambda x: float(geohash.decode(x)[0]))
data_df['geo_y'] = data_df['geo_code'].apply(lambda x: float(geohash.decode(x)[1]))
data_df['geo1'] = data_df['geo_code'].apply(lambda x: x[:2])
data_df['geo2'] = data_df['geo_code'].apply(lambda x: x[:1])
# data_df.drop(columns=['geo_code'],inplace=True)

# convert some cols to str type
data_df['channel'] = data_df['channel'].fillna('N/A').astype(str)
data_df['trans_type2'] = data_df['trans_type2'].fillna('N/A').astype(str)
data_df['market_type'] = data_df['market_type'].fillna('N/A').astype(str)

exclude = ['UID','time','day','trans_amt','bal','in_op']
for c in data_df.columns:
    if 'last' in c or 'next' in c:
        exclude.append(c)

print('clean rare...')
specific_limit = {}
specific_limit['amt_src1'] = 100
specific_limit['merchant'] = 100
specific_limit['trans_type1'] = 500
specific_limit['device1'] = 100
specific_limit['device2'] = 500
specific_limit['mac1'] = 1000
specific_limit['amt_src2'] = 1000
specific_limit['geo2'] = 10
specific_limit['geo1'] = 100
specific_limit['geo_code'] = 500
specific_limit['ip1_sub'] = 100

keep_value = {}
for c in utils.tran_variable.tran_important_columns:
    keep_value[c] = utils.tran_variable.care_about[c]

RC = rare_cleaner.RareCleaner(columns=[], limit=100, specific_limit=specific_limit,keep_value=keep_value, drop_nan=False, display=True)
RC.fit(data_df)
RC.transform(data_df)

mix_col_tuple = [
['geo_code','merchant'],
['device1','geo_y'],
['merchant','next_same_amt_src1'],
['amt_src2','merchant'],
['channel','device1'],
['device1','mac1'],
['next_same_device1','merchant'],
['geo_x','last_same_trans_type2'],
['merchant','device1'],
['device1','geo_code'],
['merchant','ip1_sub'],
['trans_type1','merchant'],
['geo_x','device2'],
['amt_src1','amt_src2'],
['trans_type1','trans_type2'],
]
cols = ['channel','trans_type2','market_type',]
mix_cols = []
for c1,c2 in utils.shuffle_cols(cols)+mix_col_tuple:
    if '{}.{}'.format(c1,c2) not in mix_cols:
        mix_cols.append('{}.{}'.format(c1,c2))
        data_df['{}.{}'.format(c1,c2)] = data_df[c1].fillna('N/A').astype(str) + '.' + data_df[c2].fillna('N/A').astype(str)

print('处理 time...')
data_df['h'] = data_df['time'].apply(lambda x: int(x[:2]))
data_df['m'] = data_df['time'].apply(lambda x: int(x[3:5]))
data_df['s'] = data_df['time'].apply(lambda x: int(x[6:]))
data_df['time_period'] = data_df['h'].apply(utils.hour_period)
data_df['min_last5'] = (data_df['m']-55)>=0
data_df['hour_last'] = (data_df['h']-23)>=0
data_df['sec_last10'] = (data_df['s']-50)>=0
data_df.drop(columns=['time'],inplace=True)

print('处理day')
data_df['day_last5'] = (data_df['day']-25)>=0
# data_df.drop(columns=['day'],inplace=True)

print('process money...')
data_df['trans_amt'].fillna(0,inplace=True)
data_df['bal'].fillna(0,inplace=True)
data_df['total_money'] = data_df['trans_amt'] + data_df['bal']
data_df['spend_ratio'] = data_df['trans_amt']/data_df['bal']

exclude.extend(['h','m','s','time_period','min_last5','hour_last','sec_last10','day_last5'])
print('one-hot-encode')
# 将某些列的某些值加入onehot
one_hot_cols = []
for c in utils.tran_variable.tran_important_columns:
    if len(utils.tran_variable.care_about[c])>0:
        data_df[c+'_important'] = data_df[c].apply(lambda x: x if x in utils.tran_variable.care_about[c] else 'other')
        one_hot_cols.append(c+'_important')

one_hot_cols.extend(mix_cols+['geo1','geo2'])
for c in one_hot_cols:
    if len(data_df[c].unique())<=30:
        data_df[c] = data_df[c].fillna('N/A').astype(str)
        print('\t{:<15}:{:<8}'.format(c,len(data_df[c].unique())))
        append_column = pd.get_dummies(data_df[c])
        append_column = append_column.rename(columns = lambda x:c+'_'+str(x))   #改名,避免列名冲突
        data_df = pd.concat([data_df,append_column],axis=1)
    else:
        one_hot_cols.remove(c)
data_df.drop(columns=one_hot_cols,inplace=True)
print('label-encode')
for c in data_df.columns:
    if (data_df[c].dtype=='object') and (c not in one_hot_cols) and (c not in exclude):
        print('\t{:<15}:{:<8}'.format(c, len(data_df[c].unique())))
        lbl = LabelEncoder()
        lbl.fit(list(data_df[c].values.astype('str')))
        data_df[c] = lbl.transform(list(data_df[c].values.astype('str')))

assert train_num+test_num == data_df.shape[0]
train_df = data_df[:train_num]
test_df = data_df[train_num:]

train_df = pd.merge(train_df,tag,on='UID')
train_label = train_df['Tag'].values
train_id = train_df['UID'].values
train_set = train_df.drop(['UID','Tag'],axis=1).values
feature_names = train_df.drop(['UID','Tag'],axis=1).columns
test_id = test_df['UID'].values
test_set = test_df.drop(['UID'],axis=1).values
print(train_label.shape)
print(train_set.shape)
print(test_id.shape)
print(test_set.shape)

import xgboost as xgb
params = {
    'tree_method': 'gpu_hist',
    'gpu_id': 1,
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'gamma': 0.1,
    'max_depth': 7,
    'subsample': 0.8,              # 随机采样训练样本
    'colsample_bytree': 0.8,       # 生成树时进行的列采样
    'min_child_weight': 2,
    'silent': 1,
    'eta': 0.1,
    'alpha':1e-5,
    'lambda':1e-5,
    'seed': 1000,
    'scale_pos_weight':1,
    'nthread': 4,
}

dtrain = xgb.DMatrix(train_set, label=train_label,feature_names=feature_names)
num_round = 20
bst = xgb.train(params, dtrain, num_round)
print('Done')

# test prob
dtest = xgb.DMatrix(test_set,feature_names=feature_names)
pred = bst.predict(dtest)
pred_df = pd.DataFrame({'UID':test_id,'Tag':pred})
group = pred_df.groupby('UID')
prob_info_test = group.apply(utils.extract_prob,file='tr').reset_index()
prob_info_test.to_csv(os.path.join(file_path,'tran_prob_test.csv'))

# train prob
dtest = xgb.DMatrix(train_set,feature_names=feature_names)
pred = bst.predict(dtest)
pred_df = pd.DataFrame({'UID':train_id,'Tag':pred})
group = pred_df.groupby('UID')
prob_info_train = group.apply(utils.extract_prob,file='tr').reset_index()
prob_info_train.to_csv(os.path.join(file_path,'tran_prob_train.csv'))
print('Done')


dtrain = xgb.DMatrix(train_set, label=train_label,feature_names=feature_names)
num_round = 3000
bst = xgb.train(params, dtrain, num_round)
print('Done')
# test prob
dtest = xgb.DMatrix(test_set,feature_names=feature_names)
pred = bst.predict(dtest)
pred_df = pd.DataFrame({'UID':test_id,'Tag':pred})
pred_df.to_csv(os.path.join(file_path,'tran_prob_TestPredict.csv'))