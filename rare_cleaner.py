import pandas as pd
import numpy as np

'''
对于df里面的每列,将存在数量相遇limit的值替换成fill
columns：list 要处理的列
limit:int
specific:dict 特殊limit的列
drop_nan:bool 是否保留nan
fill:填充值
display：bool 是否显示处理前后value的数量变化
'''
class RareCleaner():
    def __init__(self, columns, limit, specific_limit={}, drop_nan=True, fill='other',display=False):
        self.columns = columns
        self.limit = limit
        self.drop_nan = drop_nan
        self.fill = fill
        self.specific_limit = specific_limit
        self.save_values = {}
        self.drop_values = {}
        self.display = display

    def fit(self, df):
        for c in self.columns:
            assert c in df.columns
            if c in self.specific_limit.keys():
                limit = self.specific_limit[c]
            else:
                limit = self.limit
            value_count = df[c].value_counts(dropna=self.drop_nan)
            save_value = value_count >= limit
            save_value = set(value_count.index[save_value].values)
            drop_value = value_count < limit
            drop_value = set(value_count.index[drop_value].values)

            self.save_values[c] = save_value
            self.drop_values[c] = drop_value

    def ifSave(self, x, save_set):
        if pd.isnull(x):
            if pd.isnull(list(save_set)).any():
                return np.nan
            else:
                return self.fill
        else:
            if x in save_set:
                return x
            else:
                return self.fill

    def transform(self, df):
        if self.display:
            print('begin transform...')
        for c in self.columns:
            assert c in df.columns
            before_num = len(df[c].unique())
            df[c] = df[c].apply(lambda x: self.ifSave(x, self.save_values[c]))
            if self.display:
                print('\tprocess {} ...\tbefore_num:{}\tafter_num:{}'.format(c,before_num,len(df[c].unique())))