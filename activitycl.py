import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,MiniBatchKMeans,SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import operator
import csv

"""def purity(a,b,numofclusters):
    index = 0
    df = {}
    labeldist = Counter(b)
    column = range(numofclusters)
    sum = 0
    for i in column:
        df.update({i: np.array([])})
    for val in a:
        df[val] = np.append(df[val],b[index])
        index += 1
    for i in column:
        cnt = Counter(list(df[i]))
        bestkey = 0
        bestres = 0
        for key in cnt:
            if float(cnt[key])/labeldist[key] > bestres:
                bestkey = key
                bestres = float(cnt[bestkey])/labeldist[bestkey]
        sum += cnt[bestkey]
    purti = float(sum)/len(a)
    return purti
"""

def purity(activities,cluster):
    uniact = np.unique(activities)
    unicl = np.unique(cluster)
    p = {}
    c = {}
    final = {}
    for cl in unicl:
        temp = []
        for i,val in enumerate(cluster):
            if val == cl:
                temp.append(activities[i])
        cnt = Counter(temp)
        c.update({cl:cnt})
        dic ={}
        for act in uniact:
            dic.update({act : cnt[act]/float(len(temp))})
        p.update({cl: dic})
    for a in uniact:
        num = 0
        den = 0
        for ccl in unicl:
            num += p[ccl][a]*c[ccl][a]
            den += c[ccl][a]
        final.update({a: num/float(den)})
    return final

def clusterdata(cluster,n_cluster):
    kmeans = KMeans(n_clusters = n_cluster).fit(cluster.reshape(-1,1))
    cluster_labels = kmeans.fit_predict(cluster.reshape(-1,1))
    #silhouette_avg = silhouette_score(test, cluster_labels)
    return cluster_labels

def actcl(data,nm):
    part = data['participant']
    groups = data['combined']
    act = data['activity']
#    group_kfold = GroupKFold(n_splits = 2)
#    one, two = group_kfold.split(data,range(len(data)),groups)
#    dataone = data.loc[one[1]]
#    datatwo = data.loc[two[1]]
#    acttwo = list(datatwo['activity'])
#    actone = list(dataone['activity'])
    uniact = np.unique(data['activity'])
    del data['combined']
    del data['participant']
    del data['activity']
#    del datatwo['combined']
#    del datatwo['participant']
#    del datatwo['activity']
    dictnum = {}
    dictkey = {}
    dictcl = {}
    for val in uniact:
        dictnum.update({val:0})
        dictkey.update({val:'NA'})
        dictcl.update({val:'NA'})
    #with open('clfs.csv','w') as csvfile:
    #write = csv.writer(csvfile,delimiter=',')
    index = 1
    retfinal = np.zeros([16,35])
    for col in data:
        print col
        tocluster = data[col].as_matrix()
#        totest = datatwo[col].as_matrix()
        for i in [16]:
            clusterone = clusterdata(tocluster,i)
#            clustertwo = clusterdata(totest,tocluster,i)
#            act = np.concatenate((acttwo, actone),axis = 0)
#            cluster = np.concatenate((clusterone,clustertwo),axis = 0)
            p = purity(act,clusterone)
#            temp = np.array([])
#            if index == 1:
#                label = []
#                for ac in p:
#                    label.append(ac)
#                    temp = np.append(temp,p[ac])
#            else:
#                for ac in label:
#                    temp = np.append(temp,p[ac])
    
#            if index == 1:
#                pri = temp
#            else:
#                pri = np.append(pri,[temp], axis = 0)
            index+=1
            if 'Unnamed' not in col:
                for key, value in dictnum.iteritems():
                    if p[key] > dictnum[key]:
                        dictnum[key] = p[key]
                        dictkey[key] = col
                        dictcl[key] = i
#    with open('dict.csv', 'wb') as csv_file:
#        writer = csv.writer(csv_file)
#        for key, value in dictkey.items():
#            writer.writerow([key, value])
    
    print dictnum
    print dictkey
    print dictcl
    return np.unique(dictkey.values())
            #    write.writerow([col, i, max(p.iteritems(),key = operator.itemgetter(1))[0], max(p.values())])
    
#        
#actcl()
