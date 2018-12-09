import pandas as pd
import numpy as np
from functools import reduce
import rare_cleaner
import os
import utils

'''
oper type1 改成全部列
只保留both value x

!!! 需要重新筛选重要列,只留both value
'''

nrows = None
ssh = True

if ssh:
    file_path = '/data1/jianglibin/orange/data/'
else:
    file_path = 'data'

train_tr = pd.read_csv(os.path.join(file_path,'transaction_train_new.csv'),nrows = nrows)
test_tr = pd.read_csv(os.path.join(file_path,'transaction_round1_new.csv'),nrows = nrows)

train_uid = train_tr['UID'].values
test_uid = test_tr['UID'].values

data_tr = pd.concat([train_tr,test_tr],axis=0).reset_index(drop=True)

print(train_tr.shape)
print(test_tr.shape)
print(data_tr.shape)
print(len(data_tr['UID'].unique()))

merge_list = []
be_processed = {}
original_columns = list(data_tr.columns)
original_columns.remove('UID')
for c in data_tr.columns:
    be_processed[c] = False

print('process decive2...')
data_tr['device2'] = data_tr['device2'].apply(lambda x: utils.convert_device2(x) if pd.notnull(x) else x)

# print('keep both value...')
# data_tr = utils.save_both_value(data_tr,utils.tran_variable.tran_both_value)

print('process geo_code...')
data_tr['geo1'] = data_tr['geo_code'].fillna('0000').apply(lambda x: x[:2])
data_tr['geo2'] = data_tr['geo_code'].fillna('0000').apply(lambda x: x[:1])


print('处理 类型1 cols...')
# 类型1: 这部分cols含有较多的值，如果对每个值的统计信息都记录会是的最终feature非常多，所以只保留部分统计信息
# 注意:参加类型1的cols也可以参加类型2
#      这些cols的统计应该在clean rare之前,因为clean rare会带来次数统计的不真实
# type_columns = ['geo_code', 'device1', 'device2', 'merchant', 'code1', 'code2', 'acc_id1', 'acc_id2', 'acc_id3',
#                 'device_code1', 'device_code2', 'device_code3', 'mac1', 'ip1', 'ip1_sub','market_code']
for c in data_tr.columns:
    if c not in ['UID','day','time','trans_amt','bal']:
        print('\t', c)
        be_processed[c] = True
        group = data_tr[['UID', c]].groupby('UID')
        c_info = group.apply(lambda x:utils.extract_type1(x,c,file='tr')).reset_index()
        merge_list.append(c_info)

print('clean rare columns...')
specific_limit = {}
for c in utils.tran_variable.tran_important_columns:
    d = data_tr[c].value_counts(dropna=False)
    if d.values.shape[0] > 100:
        specific_limit[c] = d.values[100]
    else:
        specific_limit[c] = d.values[-1]

keep_value = {}
for c in utils.tran_variable.tran_important_columns:
    keep_value[c] = utils.tran_variable.care_about[c]

RC = rare_cleaner.RareCleaner(columns=[], limit=100, specific_limit=specific_limit,keep_value=keep_value, drop_nan=False, display=True)
RC.fit(data_tr)
RC.transform(data_tr)

# convert back
data_tr['geo1'] = data_tr['geo1'].apply(lambda x: '11' if x == 'other' else x)
data_tr['geo2'] = data_tr['geo2'].apply(lambda x: '1' if x == 'other' else x)


print('处理 类型2 的cols...')
# 类型2: 这部分col处理后只剩下少量值的统计信息，分别记录这些值的出现次数和频率
# 注意，参加类型2的cols也可以参加类型1
# 对于在utils.tran_important_columns内的cols中只保留'care about'
# columns = ['channel', 'geo_fillna', 'geo1', 'geo2', 'market_type', 'trans_type1', 'trans_type2',
#            'device1', 'device2', 'amt_src1', 'amt_src2', 'merchant', 'mac1', 'ip1_sub',]
columns = utils.tran_variable.tran_important_columns
data_tr[columns] = data_tr[columns].fillna('N/A').astype(str)
# extract info
for c in columns:
    print('\t', c)
    be_processed[c] = True
    tr_rename_c = {}
    for name in data_tr[c].unique():
        tr_rename_c[name] = 'tr_{}_{}'.format(str(c),str(name))
    c_info = pd.pivot_table(data_tr[['UID', c]], index='UID', columns=c, aggfunc=len, fill_value=0).reset_index()
    c_info = c_info.rename(columns=tr_rename_c)
    sum = c_info.drop('UID', axis=1).sum(axis=1)
    for col in c_info.columns[1:]:
        c_info[str(col) + '_freq'] = (c_info[col] / sum).astype(float)

    c_info = c_info[utils.tran_variable.care_about[c+'_columns']]

    merge_list.append(c_info)

print('处理金额...')
data_tr['total_money'] = data_tr['trans_amt'] + data_tr['bal']
data_tr['spend_ratio'] = data_tr['trans_amt'] / data_tr['bal']
columns = ['trans_amt', 'bal', 'total_money','spend_ratio']
for c in columns:
    print('\t',c)
    be_processed[c] = True
    tr_sum = data_tr[['UID', c]].groupby('UID')[c].sum().reset_index(name='tr_' + c + '_sum')
    tr_mean = data_tr[['UID', c]].groupby('UID')[c].mean().reset_index(name='tr_' + c + '_mean')
    tr_std = data_tr[['UID', c]].groupby('UID')[c].std().reset_index(name='tr_' + c + '_std')
    merge_list.extend([tr_mean, tr_sum,tr_std])

print('处理 day...')
be_processed['day'] = True
group = data_tr[['UID', 'day']].groupby('UID')
tr_day_info = group.apply(lambda x: utils.extract_day(x,file='tr')).reset_index()
tr_day_std = group['day'].std().reset_index(name='tr_day_std')
data_tr['day_last5'] = (data_tr['day']-25)>=0
tr_day_last5 = pd.pivot_table(data_tr[['UID', 'day_last5']], index='UID', columns='day_last5', aggfunc=len,fill_value=0).reset_index()
tr_day_last5 = tr_day_last5.rename(columns={False: 'tr_day_not_last5', True: 'tr_day_is_last5'})

day_info = reduce(lambda left, right: pd.merge(left, right, on='UID'),[tr_day_info, tr_day_std, tr_day_last5])
merge_list.append(day_info)

# process time
print('处理 time...')
be_processed['time'] = True
data_tr['h'] = data_tr['time'].apply(lambda x: int(x[:2]))
data_tr['m'] = data_tr['time'].apply(lambda x: int(x[3:5]))
data_tr['s'] = data_tr['time'].apply(lambda x: int(x[6:]))
data_tr['time_period'] = data_tr['h'].apply(utils.hour_period)
time_info = pd.pivot_table(data_tr[['UID', 'time_period']], index='UID', columns='time_period', aggfunc=len,fill_value=0).reset_index()
time_info = time_info.rename(columns={0: 'tr_time_period_0', 1: 'tr_time_period_1', 2: 'tr_time_period_2', 3: 'tr_time_period_3'})
for c in time_info.columns[1:]:
    time_info[c + '_freq'] = time_info[c] / time_info[time_info.columns[1:]].sum(axis=1)

data_tr['minute_last5'] = (data_tr['m']-55)>=0
minute_info = pd.pivot_table(data_tr[['UID', 'minute_last5']], index='UID', columns='minute_last5', aggfunc=len,fill_value=0).reset_index()
minute_info = minute_info.rename(columns={False: 'tr_time_minute_notlast5', True: 'tr_time_minute_last5'})
minute_info['tr_time_minute_last5_freq'] = minute_info['tr_time_minute_last5'] / minute_info[minute_info.columns[1:]].sum(axis=1)
minute_info = minute_info[['UID', 'tr_time_minute_last5', 'tr_time_minute_last5_freq']]

data_tr['hour_last1'] = (data_tr['h']-23)>=0
hour_info = pd.pivot_table(data_tr[['UID', 'hour_last1']], index='UID', columns='hour_last1', aggfunc=len,fill_value=0).reset_index()
hour_info = hour_info.rename(columns={False: 'tr_time_hour_notlast1', True: 'tr_time_hour_last1'})
hour_info['tr_time_hour_last1_freq'] = hour_info['tr_time_hour_last1'] / hour_info[hour_info.columns[1:]].sum(axis=1)
hour_info = hour_info[['UID', 'tr_time_hour_last1', 'tr_time_hour_last1_freq']]

data_tr['second_last10'] = (data_tr['s']-50)>=0
second_info = pd.pivot_table(data_tr[['UID', 'second_last10']], index='UID', columns='second_last10', aggfunc=len,fill_value=0).reset_index()
second_info = second_info.rename(columns={False: 'tr_time_second_notlast10', True: 'tr_time_second_last10'})
second_info['tr_time_second_last10_freq'] = second_info['tr_time_second_last10'] / second_info[second_info.columns[1:]].sum(axis=1)
second_info = second_info[['UID', 'tr_time_second_last10', 'tr_time_second_last10_freq']]
merge_list.extend([time_info, minute_info,hour_info,second_info])

print('Pre-Preocess Log:')
for c in original_columns:
    if not be_processed[c]:
        print("Column {} haven't be processed".format(c))

tr_final = reduce(lambda left, right: pd.merge(left, right, on='UID'), merge_list)

train_tran = tr_final[tr_final['UID'].apply(lambda x: x in train_uid)]
test_tran = tr_final[tr_final['UID'].apply(lambda x: x in test_uid)]

print(tr_final.shape)
print(train_tran.shape)
print(test_tran.shape)

train_tran.to_csv(os.path.join(file_path,'train_tran_cleanrare.csv'))
test_tran.to_csv(os.path.join(file_path,'test_tran_cleanrare.csv'))
