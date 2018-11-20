import pandas as pd
import numpy as np
from IPython.display import display, HTML
from functools import reduce
import rare_cleaner


train_df = pd.DataFrame()
test_df = pd.DataFrame
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

# 该列trainset和testset中前n多的值的交集
c = ''
n = 20
train_top_index = set(train_df[c].value_counts().index[:n])
test_top_index = set(test_df[c].value_counts().index[:n])
print(len(train_top_index & test_top_index))
print(train_top_index & test_top_index)


# 该列中需要比较关心的值,
c = ''
bad_top5 = train_df[train_df['Tag']==1][c].value_counts(dropna=False).index[:5]
bad_top10 = train_df[train_df['Tag']==1][c].value_counts(dropna=False).index[:10]
good_top5 = train_df[train_df['Tag']==0][c].value_counts(dropna=False).index[:5]
good_top10 = train_df[train_df['Tag']==0][c].value_counts(dropna=False).index[:10]

differ1 = set(bad_top10).difference(set(good_top10))
differ2 = set(good_top10).difference(set(bad_top10))

print(bad_top5)
print(good_top5)
print(differ1)
print(differ2)
print('-----------')
total = list(bad_top5)+list(good_top5)+list(differ1)+list(differ2)
for c in set(total):
    print("'{}',".format(c))

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
