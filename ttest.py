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
        colorspart = cmap(np.linspace(0., 1., len(parti)))
        colorslab = cmap(np.linspace(0., 1., len(labi)))
        colorsact = cmap(np.linspace(0., 1., len(acti)))
        #plt.pie(parti.values(),labels=parti.keys(),autopct=make_autopct(parti.values()),shadow=True)
        #plt.show()
        plt.pie(labi.values(),labels=labi.keys(),autopct=make_autopct(labi.values()),shadow=True)
        plt.show()
        plt.pie(acti.values(),labels=acti.keys(),autopct=make_autopct(acti.values()),shadow=True)
        plt.show()
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
        

def cluster_analysis(cluster, part_label, act_label, def_label,definitions,index):
    deflabelmin = []
    for i in range(len(cluster)):
        deflabelmin.append(findlabel(part_label[i],act_label[i],def_label,definitions))
    ccluster = convert(cluster,deflabelmin)
    return cohen_kappa_score(ccluster,deflabelmin),v_measure_score(cluster,deflabelmin),adjusted_rand_score(cluster,deflabelmin),purity(cluster,deflabelmin,index),normalized_mutual_info_score(ccluster,deflabelmin),deflabelmin




def clusterdata(cluster,n_cluster):
    kmeans = AgglomerativeClustering(n_clusters = n_cluster).fit(cluster)
    cluster_labels = kmeans.fit_predict(cluster)
    silhouette_avg = silhouette_score(cluster, cluster_labels)
    return silhouette_avg, cluster_labels

ttest = pd.read_csv('ttestfeatures.csv')
act = ttest['activity']
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
    if 'Participant' in col or 'activity' in col or 'window_number' in col:
        continue
    else:
        x = ttest[col].iloc[TSST]
        y = ttest[col].iloc[Rest]
        stat, p =  stats.ttest_ind(list(x),list(y))
        if float(p) < .01:
            pvals.append(p)
            features.append(col)
            tstat.append(stat)
with open('ttestresults.csv','w') as ttestres:
    write = csv.writer(ttestres,delimiter=',')
    write.writerow(pvals)
    write.writerow(tstat)
    write.writerow(features)
ecgfeatures = [x for x in features if 'ecg' in x]
gsrfeatures = ['mean_GSR','min_GSR','sd_GSR','meidan_GSR','08per_GSR','20perc_GSR']
hrfeatures = ['min_polar','max_polar']
ecggsrfeatures = ecgfeatures + gsrfeatures
df = pd.read_csv('definitionnew.csv')
dm = df[['part','activity']]
da = dm.as_matrix()
df = df[['yesno','average','pss','consensus','Happy','Worried']]
bestcohen = 0
silhouettes = []
with open('resultsofcluster.csv','w') as csvfile:
    write = csv.writer(csvfile,delimiter=',')
    write.writerow(['features','definitions','k_clusters','cohens','v_measure','rand_index','purity','normalized_mutual_info_score','silhouette'])
    
    for i in [1,3,4]:
        silhoutteaverages = []
        purities = []
        wpurities = []
        ypurities = []
        hpurities = []
        if i == 0:
            name = 'ecg'
            cluster = ttest[ecgfeatures]
            dashline = 'r--'
        if i == 1:
            name = 'gsr'
            dashline = 'b--'
            cluster = ttest[gsrfeatures]
        if i == 2:
            dashline = 'g--'
            name = 'ecg+gsr'
            cluster = ttest[ecggsrfeatures]
        if i == 3:
            dashline = 'm--'
            name = 'HR'
            cluster = ttest[hrfeatures]
        if i == 4:
            dashline = 'y--'
            name = 'All'
            cluster = ttest[gsrfeatures+hrfeatures]
        print name
        cluster = cluster.as_matrix()
        for j in range(3,4):
            x,y = clusterdata(cluster,j)
            silhoutteaverages.append(x)
            for col in df:
                testwith = list(df[col])
                n,m,o,r,s,deflabel = cluster_analysis(y,list(ttest['Participant']),list(ttest['activity']),da,testwith,j)
                if 'pss' in col:
                    purities.append(r)
                if  r > bestcohen and 'average' in col and j == 3 and 'All' in name:
                    bestname = name
                    bestnum = j
                    bestcohen = r
                    bestlabel = y
                    bestcol = col
                    besttest = deflabel
                write.writerow([name,col,j,n,m,o,r,s,x])
        plt.plot(range(3,4),purities,dashline)
    plt.legend(handles = [mpatches.Patch(color = 'red',label = 'ECG Only'),mpatches.Patch(color = 'blue',label = 'GSR Only'),mpatches.Patch(color = 'green',label = 'GSR + ECG'),mpatches.Patch(color = 'Purple',label = 'HR'),mpatches.Patch(color = 'yellow',label = 'All')],loc = 4)
    plt.xlabel("Number of Clusters")
    plt.ylabel("Purity")
    plt.show()
print bestcol
print bestname
print bestnum
print sum(besttest)
print getpiecharts(bestlabel,besttest,list(ttest['Participant']),list(ttest['activity']),bestnum)


with open('bestlabel.csv','w') as cfile:
        write = csv.writer(cfile,delimiter=',')
        write.writerow(bestlabel)
