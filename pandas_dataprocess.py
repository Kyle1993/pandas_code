import pandas as pd
import numpy as np
from IPython.display import display, HTML
from functools import reduce
import rare_cleaner


train_df = pd.DataFrame()
test_df = pd.DataFrame
'''
jupyter 打印df
'''
from IPython.display import display, HTML
display(HTML(pd.DataFrame(train_df).to_html()))

'''
read_csv 经常会出现第一列是'unnamed 0',使用index_col去除
'''
df1 = pd.read_csv('',index_col=0)

'''
reset_index的时候要注意drop=True，不然可能造成多出一列index
'''
df2 = pd.concat([train_df,test_df],sort=False).reset_index(drop=True)


'''
show train_df
'''
# 显示各列名字及dtype
for c in train_df.columns:
    print('{}\t{}'.format(c,train_df[c].dtype))

# 显示某列基本信息
c = 'col1'
print(train_df[c].isnull().any())
print(train_df[c].dtype)
print(len(train_df[c].unique()))
print(train_df[c].unique())
d = train_df[c].value_counts(dropna=False)
display(HTML(pd.DataFrame(d).to_html()))

# 各列的前n多的值的数量在该列(trainset & testset)的占比
cols = []
n = 5
dropna = False

for c in cols:
    if not dropna:
        train_num = train_df.shape[0]
        test_num = test_df.shape[0]
    else:
        train_num = train_df[c].dropna().shape[0]
        test_num = test_df[c].dropna().shape[0]

    d = train_df[c].value_counts(dropna=dropna)
    topN_indexs = set(d.index[:n])
    topN_num_train = int(train_df[c].apply(lambda x: x in topN_indexs).sum())
    topN_num_test = int(test_df[c].apply(lambda x: x in topN_indexs).sum())
    print(topN_num_train/train_num,topN_num_test/test_num)

'''
透视表
'''
# 统计在通过 groupby('UID')操作后 c列的各个值出现次数
c = ''
c_info = pd.pivot_table(train_df[['UID', c]], index='UID', columns=c, aggfunc=len, fill_value=0).reset_index()

'''
使用apply
'''
# apply是对series内各个元素逐一使用
# 如果需要给apply内传入的自定义函数传参可以这样
def f(x,param1):
    return x
c = ''
param1 = 1
train_df[c] = train_df[c].appely(lambda x: f(x,param1))

'''
多表合并
'''
# 把merge_list内的df根据UID合并成一个表
merge_list = []
data_final = reduce(lambda left, right: pd.merge(left, right, on='UID'),merge_list)

'''
列重命名
'''
# 防止trianset和testset的同名列出现在不同的位置,不要直接用df.columns = ['a','b'..]的方式重命名
rename_dict = {'a':'new_a','b':'new_b'}
train_df.rename(columns=rename_dict)
test_df.rename(columns=rename_dict)

'''
apply返回多个值，合并成一个df
groupby 之后的apply是对每个group进行操作
一般df的apply是对column里的每个值进行操作
两者都可以使用apply返回多个值并合并成一个df
'''
pred_df = pd.DataFrame({'UID':[1,1,2,2,2],'Value':[1,2,3,4,5]})
group = pred_df.groupby('UID')
def prob_extract(g):
    v = g['Value'].values
    v_max = np.max(v)
    v_min = np.min(v)
    v_avg = np.mean(v)
    v_std = np.std(v)
    res = pd.Series({'v_max':v_max,
                     'v_min':v_min,
                     'v_avg':v_avg,
                     'v_std':v_std,
                 })
    return res
prob_info_test = group.apply(prob_extract).reset_index()


'''
如果某一列的unique过多,只能leableecnode,但是又想找一些关键值做onehot
1.先clean rare,看下值是否依然过多
2.通过一些简单指标筛选出一些关键值
3.将这些关键值onehot之后丢入xgboost训练一会,输出.get_score,查看重要性，进一步筛选
'''
# 第二步,初步筛选一些关键值
N1 = 5
N2 = 20
c = ''

bad_topN1 = train_df[train_df['Tag']==1][c].value_counts(dropna=False).index[:N1]
bad_topN2 = train_df[train_df['Tag']==1][c].value_counts(dropna=False).index[:N2]
good_topN1 = train_df[train_df['Tag']==0][c].value_counts(dropna=False).index[:N1]
good_topN2 = train_df[train_df['Tag']==0][c].value_counts(dropna=False).index[:N2]

differ1 = set(bad_topN2).difference(set(good_topN2))
differ2 = set(good_topN2).difference(set(bad_topN2))

rare_limit = 10
length = 10
data_df = train_df[[c,'Tag']].dropna()
num = data_df.shape[0]
value_num = len(data_df[c].unique())
mean_ = data_df['Tag'].mean()
c_info = pd.pivot_table(data_df, index=c, columns='Tag', aggfunc=len, fill_value=0).reset_index()  # 列c关于Tag的透视表
c_info['sum'] = c_info[0]+c_info[1]
# 去除稀少值，因为稀少值会造成倍数特别大
c_info = c_info[(c_info[1]>rare_limit) & (c_info[0]>rare_limit)]
c_info = c_info[c_info['sum']>3*(num/value_num)]
# 平衡正负样本比例
ratio = (1-mean_)/mean_
c_info[1] = c_info[1]*ratio
# 筛选value
# 用长度去限定阈值，当然也可以直接制定阈值
for i in range(5,100):
    negitive_care = c_info[c_info[0]/c_info[1]>i][c].values
    positive_care = c_info[c_info[1]/c_info[0]>i][c].values
    if len(list(negitive_care)+list(positive_care))<length:
        print(i)
        break
# negitive_care 和 positive_care 可能比较重要
important_value = set(good_topN1 + bad_topN1 + differ1 + differ2 + negitive_care + positive_care)

