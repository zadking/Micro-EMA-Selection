# -*- coding: utf-8 -*-
"""
Created on Tue May 08 22:11:37 2018

@author: zdking
"""

import pandas as pd
import csv
import numpy as np

data = pd.read_csv('question_for_correlation_matrix_normalized.csv')
df = data[['activity','participant']]
parts = np.unique(data['participant'].as_matrix())
for col in data:
    if 'activity' in col or 'participant' in col:
        continue
    temp = np.array([])
    for p in parts:
        te = data.loc[data['participant'] == p]
        vals = te[col].as_matrix()
        avg = np.average(vals)
        for v in vals:
            if v >= avg:
                temp = np.append(temp,[1])
            else:
                temp = np.append(temp,[0])
    data[col] = temp
da = pd.read_csv('data.csv')
column = list(data)
for index, row in da.iterrows():
    part = row['participant']    
    act = row['activity']
    select = data.loc[(data['participant']==part) & (data['activity'] == act)].as_matrix()
    select = select[0:13]
    if index == 0:
        fill = select
    else:
        fill = np.append(fill,select,axis = 0)
ret = pd.DataFrame(fill, columns = column)
ret.to_csv('allquestionlabels.csv')