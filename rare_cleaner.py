import pandas as pd
import numpy as np


class RareCleaner():
    '''
    对于df里面的每列,将存在数量相遇limit的值替换成fill
    columns：list 要处理的列
    limit:int
    specific:dict 特殊limit的列 {col1:limut1,col2:limit2,...}
    drop_nan:bool 是否保留nan
    fill:填充值
    display：bool 是否显示处理前后value的数量变化
    keep_value: dict 列需要被保留的值，不会被clean {col1:[value1,value2],col2:[value1],...}
    only_save: dict 列内只需要保存的值，优先级最高
    strict: bool True:clean the cols in 'only_save', save 'only save' value
                 False: clean the cols only in 'columns'
    '''
    def __init__(self, columns, limit, specific_limit={}, keep_value={}, only_save={}, drop_nan=True, fill='other', strict=False, display=False):
        self.columns = list(set(columns+list(specific_limit.keys())))
        self.limit = limit
        self.drop_nan = drop_nan
        self.fill = fill
        self.specific_limit = specific_limit
        self.save_values = {}
        self.display = display
        self.keep_values = keep_value
        self.only_save = only_save
        self.strict = strict

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

            self.save_values[c] = save_value
            if c in self.keep_values.keys():
                self.save_values[c] = (self.save_values[c] | set(self.keep_values[c]))

            if c not in self.only_save.keys():
                self.only_save[c] = self.save_values[c]

    def ifSave(self, x, save_set, only_save):
        if pd.isnull(x):
            if pd.isnull(list(save_set)).any():
                return np.nan
            else:
                return self.fill
        else:
            if (x in save_set) and (x in only_save):
                return x
            else:
                return self.fill

    def transform(self, df, ):
        if self.display:
            print('begin transform...')
        if self.strict:
            cols = self.only_save.keys()
        else:
            cols = self.columns

        for c in cols:
            assert c in df.columns
            before_num = len(df[c].unique())
            if c in self.columns:
                df[c] = df[c].apply(lambda x: self.ifSave(x,self.save_values[c],self.only_save[c]))
            else:
                df[c] = df[c].apply(lambda x: self.ifSave(x,self.only_save[c],self.only_save[c]))
            if self.display:
                print('\tprocess {:<12} \tbefore_num:{:<8}\tafter_num:{:<8}'.format(c,before_num,len(df[c].unique())))


if __name__ == '__main__':
    x = pd.DataFrame({'a':[1,1,3],'b':[2,3,4]})
    rc = RareCleaner(columns=['a','b'],limit=1,only_save={'a':[1,2],},display=True)
    rc.fit(x)
    rc.transform(x)
    print(x)
