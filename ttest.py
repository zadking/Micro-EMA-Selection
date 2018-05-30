import pandas as pd
import csv
import numpy as np
from scipy import stats
from sklearn.cluster import KMeans,AgglomerativeClustering,Birch,MiniBatchKMeans,SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.metrics import cohen_kappa_score,v_measure_score,adjusted_rand_score,f1_score,accuracy_score,normalized_mutual_info_score
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from activitycl import *
from sklearn.preprocessing import MaxAbsScaler


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val = int(round(pct*total/100.0))
        return '{p:.2f}% ({v:d})'.format(p=pct,v=val)
    return my_autopct


def getpiecharts(cluster, label, participants, activities, numofclusters):
    index = 0
    print cluster
    print sum(label)
    partdf = {}
    labdf = {}
    actdf = {}
    column = range(len(np.unique(cluster)))
    for i in column:
        partdf.update({i: np.array([])})
        labdf.update({i: np.array([])})
        actdf.update({i: np.array([])})
    for val in cluster:
        partdf[val] = np.append(partdf[val],participants[index])
        labdf[val] = np.append(labdf[val],label[index])
        actdf[val] = np.append(actdf[val],activities[index])
        index += 1
    cmap = plt.cm.prism
    for i in column:
        parti = Counter(partdf[i])
        labi = Counter(labdf[i])
        acti = Counter(actdf[i])
        print acti
        print labi.values()
        print labi.keys()
        colorspart = cmap(np.linspace(0., 1., len(parti)))
        colorslab = cmap(np.linspace(0., 1., len(labi)))
        colorsact = cmap(np.linspace(0., 1., len(acti)))
        #plt.pie(parti.values(),labels=parti.keys(),autopct=make_autopct(parti.values()),shadow=True)
        #plt.show()
#        plt.pie(labi.values(),labels=labi.keys(),autopct=make_autopct(labi.values()),shadow=True)
#        plt.show()
#        plt.pie(acti.values(),labels=acti.keys(),autopct=make_autopct(acti.values()),shadow=True)
#        plt.show()
    return 1

    

def findlabel(part,act,data,definitions):
    index = 0
    for val in data:
        if val[0] == part:
            if val[1] in act:
                return definitions[index]
        index += 1
    print [part,act]

def purity(a,b,numofclusters):
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

def actpurity(activities,cluster):
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
    res = sum(final.values())/float(len(final.values()))
    return res

def actclusterprec(data, part, groups, act):
        uniact = np.unique(act)
        dictnum = {}
        dictkey = {}
        dictcl = {}
        for val in uniact:
            dictnum.update({val:0})
            dictkey.update({val:'NA'})
            dictcl.update({val:'NA'})
        p = actpurity(act,data)
#        if 'Unnamed' not in col:
#            for key, value in dictnum.iteritems():
#                if p[key] > dictnum[key]:
#                    dictnum[key] = p[key]
#                    dictkey[key] = col
#                    dictcl[key] = i
        return p

def convert(cluster, ground):
    index = 0
    df = {}
    column = range(len(np.unique(cluster)))
    for i in column:
        df.update({i: np.array([])})
    for val in cluster:
        df[val] = np.append(df[val],ground[index])
        index += 1
    converted = []
    for i in cluster:
        cnt = Counter(list(df[i]))
        bkey = max(cnt, key = cnt.get)
        converted.append(bkey)
    return converted
        

def cluster_analysis(cluster, part_label, act_label, deflabelmin,definitions,index):
#    deflabelmin = []
#    for i in range(len(cluster)):
#        deflabelmin.append(findlabel(part_label[i],act_label[i],def_label,definitions))
    ccluster = convert(cluster,definitions)
    return cohen_kappa_score(ccluster,definitions),v_measure_score(cluster,definitions),adjusted_rand_score(cluster,definitions),purity(cluster,definitions,index),normalized_mutual_info_score(ccluster,definitions),definitions




def clusterdata(cluster,n_cluster):
    kmeans = AgglomerativeClustering(n_clusters = n_cluster).fit(cluster)
    cluster_labels = kmeans.fit_predict(cluster)
    silhouette_avg = silhouette_score(cluster, cluster_labels)
    return silhouette_avg, cluster_labels

def normalize(data, parts):
    for p in parts:
        test = data.loc[data['participant'] == p]
        for col in range(3,37):
            temp = test.iloc[:,col].as_matrix()
            mx = np.max(temp)
            mn = np.min(temp)
            res = np.array([float(x-mn)/float(mx-mn) for x in temp])
            test.iloc[:,col] = res
        data.loc[data['participant'] == p] = test
    return data

def ttestselect(act, ttest):
    index =0
    TSST = []
    Rest = []
    features = []
    pvals = []
    tstat = []
    for val in act:
        if 'Arithmetic' in val or 'Speech' in val or 'Sing' in val:
            TSST.append(index)
        if 'rest' in val or 'Eating' in val or 'Game' in val:
            Rest.append(index)
        index += 1
                
    for col in ttest:
        if 'Participant' in col or 'activity' in col or 'window_num' in col:
            continue
        else:
            x = ttest[col].iloc[TSST]
            y = ttest[col].iloc[Rest]
            stat, p =  stats.ttest_ind(list(x),list(y))
            if float(p) < .01:
                pvals.append(p)
                features.append(col)
                tstat.append(stat)
    return features

data = pd.read_csv('data.csv')
parts = np.unique(data['participant'].as_matrix())
data = normalize(data,parts)
#scaler = MaxAbsScaler()
#tdata = data.iloc[:,3:36]
#scaler.fit(tdata)
#tdata = scaler.transform(tdata)
#data.iloc[:,3:36] = tdata
#act = ttest['activity']
#features = ttestselect(act,ttest)
#ecgfeatures = [x for x in features if 'ecg' in x]
#gsrfeatures = ['mean_GSR','min_GSR','sd_GSR','meidan_GSR','08per_GSR','20perc_GSR']
#hrfeatures = ['min_polar','max_polar']
#ecggsrfeatures = ecgfeatures + gsrfeatures
indexname = list(data.columns)
dm = data[['participant','activity']]
da = dm.as_matrix()
df = data[['pss',	'Intended',	'BinaryStress',	'LikertStress',	'PSS1'	,'PSS2',	'PSS3',	'PSS4',	'happy',	'exicted',	'content',	'worried',	'irritable/angry',	'Sad']]
combined = data['combined']
bestcohen = 0
silhouettes = []
#with open('resultsofcluster.csv','w') as csvfile:
#    write = csv.writer(csvfile,delimiter=',')
#    write.writerow(['features','definitions','k_clusters','cohens','v_measure','rand_index','purity','normalized_mutual_info_score','silhouette'])
#    
for i in range(5):
    silhoutteaverages = []
    cohaverages = []
    purities = []
    ppurities = []
    wpurities = []
    ypurities = []
    hpurities = []
    ipurities = []
    psspurities = []
    psstpurities = []
    
    if i == 0:
        name = ['ecg']
        nm = 'ecg'
        dashline = 'r--'
    if i == 1:
        name = ['GSR']
        nm = 'gsr'
        dashline = 'b--'
    if i == 2:
        dashline = 'g--'
        nm = 'ecg+gsr'
        name = ['ecg','GSR']
    if i == 3:
        dashline = 'm--'
        nm = 'Hr'
        name = ['polar']
    if i == 4:
        dashline = 'y--'
        nm = 'All'
        name = ['ecg','GSR','polar']
    features = []
    for x in indexname:
        if 'combined' in x or 'participant' in x or 'activity' in x:
            features.append(x)
        for n in name:
            if n in x:
                features.append(x)
    features = data[features]
    selectfeatures = actcl(features,nm)
    if i == 5:
        print selectfeatures
    cluster = features[selectfeatures].as_matrix()
#    for j in range(2,10):
    for j in range(2,16):
        x,y = clusterdata(cluster,j)
        part = data['participant']
        groups = data['combined']
        act = data['activity']
        p = actclusterprec(y, part, groups, act)
        silhoutteaverages.append(p)
    if i == 0:
        totalavg = silhoutteaverages
    else:
        totalavg = [x+y for x,y in zip(totalavg,silhoutteaverages)]
    print silhoutteaverages
    plt.plot(range(2,16),silhoutteaverages,dashline)
#        for col in df:
#            testwith = list(df[col])
#            n,m,o,r,s,deflabel = cluster_analysis(y,list(data['participant']),list(data['activity']),da,testwith,j)
##            if 'worried' in col:
##                cohaverages.append(n)
##                if j == 9:
##                    plt.plot(range(2,10),cohaverages,dashline)
#            if j == 3 and 'All' in nm:
#                print col + '-----------------------------\n'
#                print n,m,o,r,s
#                print getpiecharts(y,deflabel,list(data['participant']),list(data['activity']),j)
##            if 'Intended' in col:
#                ppurities.append(r)
#                if j == 3 and 'All' in nm:
#                    pbestname = nm
#                    pbestnum = j
#                    pbestcohen = r
#                    pbestlabel = y
#                    pbestcol = col
#                    pbesttest = deflabel
#            if 'average' in col:
#                print r
#                purities.append(r)
#                if j == 3 and 'All' in nm:
#                    abestname = nm
#                    abestnum = j
#                    abestcohen = r
#                    abestlabel = y
#                    abestcol = col
#                    abesttest = deflabel
#            if 'Worried' in col:
#                wpurities.append(r)
#                if j == 3 and 'All' in nm:
#                    wbestname = nm
#                    wbestnum = j
#                    wbestcohen = r
#                    wbestlabel = y
#                    wbestcol = col
#                    wbesttest = deflabel
#            if 'Happy' in col:
#                hpurities.append(r)
#                if j == 3 and 'All' in nm:
#                    hbestname = nm
#                    hbestnum = j
#                    hbestcohen = r
#                    hbestlabel = y
#                    hbestcol = col
#                    hbesttest = deflabel
#            if 'yesno' in col:
#                ypurities.append(r)
#                if j == 3 and 'All' in nm:
#                    ybestname = nm
#                    ybestnum = j
#                    ybestcohen = r
#                    ybestlabel = y
#                    ybestcol = col
#                    ybesttest = deflabel
##                r > bestcohen and 
#            if 'Intended' in col:
#                ipurities.append(r)
#                if j == 3 and 'All' in nm:
#                    ibestname = nm
#                    ibestnum = j
#                    ibestcohen = r
#                    ibestlabel = y
#                    ibestcol = col
#                    ibesttest = deflabel
#            if 'PSS4' in col:
#                psspurities.append(r)
#                if j == 3 and 'All' in nm:
#                    pssbestname = nm
#                    pssbestnum = j
#                    pssbestcohen = r
#                    ibestlabel = y
#                    ibestcol = col
#                    ibesttest = deflabel
##                r > bestcohen and 
#            if 'PSS3' in col:
#                psstpurities.append(r)
#    plt.figure(1)
#    plt.plot(range(2,10),ipurities,dashline)
#    plt.figure(2)
#    plt.plot(range(2,10),wpurities,dashline)
#    plt.figure(3)
#    plt.plot(range(2,10),hpurities,dashline)
#    plt.figure(4)
#    plt.plot(range(2,10),ypurities,dashline)
#    plt.figure(5)
#    plt.plot(range(2,10),psspurities,dashline)
#    plt.figure(6)
#    plt.plot(range(2,10),psstpurities,dashline)
#
plt.figure(1)  
finaltotal = []
i = 0
for v in totalavg:
    if i == 0:
        finaltotal.append(0)
    else:
        finaltotal.append(v - totalavg[i-1])
    i+=1
    
print finaltotal
plt.plot(range(2,16),finaltotal,'k--')
plt.legend(handles = [mpatches.Patch(color = 'red',label = 'IBI Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + IBI'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All'),mpatches.Patch(color = 'black',label = 'Average Slope')],loc = 4)
plt.xlabel("Number of Clusters")
plt.ylabel("Cluster Precision")
#plt.show()
#plt.figure(2)    
#plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Precision")
#plt.show()
#plt.figure(3)    
#plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Precision")
#plt.show()
#plt.figure(4)    
#plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Precision")
#plt.show()
#plt.figure(5)    
#plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Precision")
#plt.show()
#plt.figure(6)    
#plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
#plt.xlabel("Number of Clusters")
#plt.ylabel("Precision")
#plt.show()
#
#plt.figure(5)
#plt.plot(silhoutteaverages)    
#plt.xlabel("Number of Clusters")
#plt.ylabel("Silhouette Score")
#plt.show()
#print bestname
##print bestnum
##print sum(besttest)
#print 'Average-----------------------------\n'
#print getpiecharts(abestlabel,abesttest,list(data['participant']),list(data['activity']),abestnum)
#print 'pss-----------------------------\n'
#print getpiecharts(pbestlabel,pbesttest,list(data['participant']),list(data['activity']),pbestnum)
#print 'worried-----------------------------\n'
#print getpiecharts(wbestlabel,wbesttest,list(data['participant']),list(data['activity']),wbestnum)
#print 'happy-----------------------------\n'
#print getpiecharts(hbestlabel,hbesttest,list(data['participant']),list(data['activity']),hbestnum)
#print 'yesno-----------------------------\n'
#print getpiecharts(ybestlabel,ybesttest,list(data['participant']),list(data['activity']),ybestnum)
#print 'Intended-----------------------------\n'
#print getpiecharts(ibestlabel,ibesttest,list(data['participant']),list(data['activity']),ibestnum)


#
#with open('bestlabel.csv','w') as cfile:
#        write = csv.writer(cfile,delimiter=',')
#        write.writerow(bestlabel)
