# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 15:31:33 2018

@author: zdking
"""
import pandas as pd
import os

def getPercent(df, start, end, activity):
    filllabel = df['Activity'].tolist()
    i = 0
    for index, row in df.iterrows():#itterate through the activities in the annotations file
        startwin = row['Start']
        endwin = row['End']
        winlen = endwin - startwin
        leninact = endwin - startwin
        if startwin < start:
            leninact = endwin - start
        if endwin > end:
            leninact = end - startwin
        percentact = float(leninact)/winlen
        if percentact > 0.5:
            filllabel[i] = activity
        else:
            filllabel[i] = 'NA'
        i += 1
    df['Activity'] = filllabel
    return df
    

Participants = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
dataPath = 'C:/Users/zdking/Habitslab/stress_data_process/stress_data_process/Pre_Pilot/'
Output = 'C:/Users/zdking/Habitslab/stress_data_process/stress_data_process/output/'
path = 'C:/Users/zdking/Habitslab/stress_data_process/stress_data_process/Preprocessing/Biostamp/'
i = 0

#for part in Participants:   
#    annotations = dataPath+str(part)+'/annotations.csv'
#    features = dataPath+str(part)+'/features.csv'

#    feat = pd.read_csv(features)
#    feat['Participant'] = part
#    feat['Activity'] = 'NA'
for f in ['Allfeatures.csv','GSRfeatures.csv','HRfeatures.csv','ECGfeatures.csv']:
    print f
    if os.path.isfile(path+ 'annotations.csv'):
        an = pd.read_csv(path+ 'annotations.csv')
    feat = pd.read_csv(path+f)
    feat['Activity'] = 'NA'
    i = 0
    for index, r in an.iterrows():#itterate through the activities in the annotations file
        if index == 266:
            print 'waint'
        startact = r['Start Timestamp (ms)']
        endact = r['Stop Timestamp (ms)']
        activity = r['EventType']
        actecg = feat.loc[(feat['End'] >= startact) & (feat['Start'] <= endact)]
        actecg = getPercent(actecg,startact, endact, activity)
        feat.loc[(feat['End'] >= startact) & (feat['Start'] <= endact)] = actecg
    if i == 0:
        returndf = feat
    else:
        returndf = returndf.append(feat)
    i += 1
    returndf.to_csv('labels' + f,index = False)