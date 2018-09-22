# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 22:25:54 2018

@author: zdking
"""
import pandas as pd


ECGonly = pd.read_csv('featureswithlabels.csv')
srdata = pd.read_csv('labels_unprocessed.csv')
ECGonly = ECGonly.dropna()
labels = ['pss-q4',	'Intended',	'BinaryStress',	'LikertStress',	'PSS1'	,'PSS2',	'PSS3',
          'PSS4',	'happy',	'excited',	'content',	'worried',	'irritable/angry',	'Sad', 'Likert06']
for lab in labels:
    ECGonly[lab] = 2
for index, row in ECGonly.iterrows():
    part = int(row['Participant'])
    act = row['Activity']
    print [part, act, index]
    for lab in labels:
        if lab in 'Likert06':
            x = list(srdata['LikertStress'][(srdata['Participant'] == part) & (srdata['Questions'] == act)])
        else:
            x = list(srdata[lab][(srdata['Participant'] == part) & (srdata['Questions'] == act)])
        if len(x) > 0:
            if 'Likert06' in lab:
                ECGonly[lab].ix[index] = x[0]
                
                
            elif 'BinaryStress' in lab:
                if x[0] in 'N':
                    ECGonly[lab].ix[index] = 0
                elif x[0] in 'no':
                    ECGonly[lab].ix[index] = 0
                else:
                    ECGonly[lab].ix[index] = 1
            elif 'pss-q4' in lab:
                thresh = list(srdata['PSS-Threshold'][(srdata['Participant'] == part) & (srdata['Questions'] == act)])[0]
                if x[0] >= thresh:
                    ECGonly[lab].ix[index] = 1
                else:
                    ECGonly[lab].ix[index] = 0
            elif 'Intended' in lab:
                ECGonly[lab].ix[index] = x[0]
            else:
                if x[0] > srdata[lab][srdata['Participant'] == part].mean():
                    ECGonly[lab].ix[index] = 1
                else:
                    ECGonly[lab].ix[index] = 0
                    
ECGonly.to_csv('ret.csv')         
print 'yes'
#        row[lab] = 