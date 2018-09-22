import csv
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd
from sklearn.metrics import cohen_kappa_score,v_measure_score,adjusted_rand_score,f1_score,accuracy_score

def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result

def findlabel(part,act,data):
    index = 0
    for val in data:
        if val[0] in part:
            if val[1] in act:
                return index
        index += 1
    return index

def cohens(adata,bdata,labela,labelb,iterations):
    cohens = {}
    for trua in range(iterations):
        trutha = trua+1
        x = {}
        for trub in range(iterations):
            truthb = trub+1
            ia = 0
            a = 0
            b = 0
            c = 0
            d = 0
            for val in labela:
                ib = findlabel(val[0],val[1],labelb)
                if adata[ia] == trutha:
                    if bdata[ib] == truthb:
                        a += 1
                    else:
                        b += 1
                else:
                    #print b[ib]
                    if bdata[ib] == truthb:
                        c += 1
                    else:
                        d += 1
                ia += 1
            total = a+b+c+d
            probo = (a + d) / float(total)
            probyes = ((a + b)/float(total)) *((a + c) / float(total))
            probno = ((c + d)/float(total)) *((b + d) / float(total))
            x.update({truthb:(probo - (probyes + probno))/(1-(probyes + probno))})
        cohens.update({trutha:x})
    return cohens

def forsk(adata,bdata,labela,labelb,iterations):
    testwit = []
    print len(labela)
    for val in labela:
        ib = findlabel(val[0],val[1],labelb)
        testwit.append(bdata[ib])
    return cohen_kappa_score(adata,testwit),v_measure_score(adata,testwit),adjusted_rand_score(adata,testwit),f1_score(adata,testwit),accuracy_score(adata,testwit)
    
    
ECGonly = pd.read_csv('ret.csv')
ECGonly = ECGonly.dropna()
data = ECGonly[ECGonly.columns[3:33]]
for part in np.unique(list(ECGonly['Participant'])):
    data[ECGonly['Participant'] == part] = normalize(data[ECGonly['Participant'] == part])

#data = numpy.loadtxt(open("featuresforclustering.csv", "rb"), delimiter=",", skiprows=1)
#sel = VarianceThreshold(threshold=(.95 * (1 - .95)))
#X = sel.fit_transform(data)
X = data
max = 0
maxind = 0
maxcluster = []
maxpca = 0
maxindpca = 0
maxclusterpca = []
for i in range(2,12):
    kmeans = KMeans(n_clusters=i, random_state=0).fit(X)
    cluster_labels = kmeans.fit_predict(X)
    silhouette_avg = silhouette_score(X, cluster_labels)
    if max < silhouette_avg:
            max = silhouette_avg
            maxind = i
            maxcluster = cluster_labels

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    #pca = PCA(n_components=i)
    #pcadata = pca.fit_transform(data)
    
    #for j in range(2,12):
    #   kmeans = KMeans(n_clusters=j, random_state=0).fit(pcadata)
    #   cluster_labels = kmeans.fit_predict(pcadata)
    #   silhouette_avg = silhouette_score(pcadata, cluster_labels)
    #   if maxpca < silhouette_avg:
    # maxpca = silhouette_avg
    #       pcaclust = i
    #       maxindpca = j
    #       maxclusterpca = [cluster_labels]
#print [pcaclust,maxindpca,maxpca]


self_report = np.loadtxt(open("srdata.csv", "rb"), delimiter=",")
kmeansself = KMeans(n_clusters=maxind, random_state=0).fit(X)
cluster_labelsself = kmeansself.fit_predict(self_report)
print [len(cluster_labelsself),len(cluster_labels)]
silhouette_avg = silhouette_score(self_report, cluster_labelsself)
with open("featurelabel.csv", "rb") as flabels:
    flabels = list(csv.reader(flabels,delimiter=","))
with open("srlabel.csv", "rb") as srlabels:
    srlabels = list(csv.reader(srlabels,delimiter=","))
df = pd.read_csv('featureselectionlabels.csv')
da = pd.read_csv('featureselectionlabels.csv')
with open('resultscohen.csv','w') as csvfile:
    write = csv.writer(csvfile,delimiter=',')
    for col in df:
        cluster = df[col]
        for ccol in da:
            flipcluster = []
            print col
            print ccol
            cluster_labelsself = da[ccol]
            if len(cluster) == 1184:
                x,y,z,t,p =  forsk(list(cluster),list(cluster_labelsself),flabels,srlabels,maxind)
                print len(list(cluster))
                for v in list(cluster):
                    if v == 2:
                        flipcluster.append(1)
                    else:
                        flipcluster.append(2)
                xf,yf,zf,tf,pf =  forsk(list(flipcluster),list(cluster_labelsself),flabels,srlabels,maxind)
                
                write.writerow([col,ccol,x,y,z,t,p,xf,yf,zf,tf,pf])
                
                #print cohens(list(cluster),list(cluster_labelsself),flabels,srlabels,maxind)
        
        
