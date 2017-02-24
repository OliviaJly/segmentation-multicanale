# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:38:37 2017

@author: Lucie
"""


########## K-means et CAH - Même méthode que SAS


# Librairies utilisées
#from scipy.spatial import distance
from scipy.cluster.vq import vq
from scipy.spatial.distance import pdist
import pandas as pd
from sklearn import cluster
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import matplotlib.pyplot as plt
import seaborn as sn
import pylab
from mpl_toolkits.mplot3d import Axes3D
import scipy.stats as spstat
import plotly
from statsmodels.graphics.mosaicplot import mosaic

# Definition du chemin où sont situées les données :
PATH = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'

## Import des données
#coordonnees dans les composantes de l'ACP
data_coor2 = pd.read_csv(PATH + '/PCA_coor2.csv', delimiter=",")

#base quanti ayant servi pour l'ACP
quanti_trans = pd.read_csv(PATH + '/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
base_test2 = quanti_trans.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', \
                                'Lecture_mess_3m', 'Ecriture_mess_3m'], 1)

#base quanti avant transformation des variables
base_quanti = pd.read_csv(PATH + '/base_quanti.csv', delimiter=",", dtype={"IDPART_CALCULE":object})
del quanti_trans

# 1er k means
#on retient les 10 premieres composantes
data_coor3 = data_coor2.iloc[:, :10]
del data_coor2

kmeans = cluster.KMeans(n_clusters=2000, max_iter=1, n_init=1) #random_state=111
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3) #affecte chaque individu au cluster le plus proche

trans = kmeans.transform(data_coor3) #donne la distance de chaque individu à chaque cluster
centers = kmeans.cluster_centers_
dist_cluster = kmeans.transform(centers) #matrice de distance entre les clusters
next_cluster = np.argsort(dist_cluster)[:, 1] #cluster le plus proche

dist_to_next_cluster = []      #distance au cluster le plus proche
for i in range(0, len(next_cluster)):
    ind = next_cluster[i]
    dist_to_next_cluster.append(dist_cluster[i, ind])
del dist_cluster, next_cluster, ind, i, trans

#compte le nb d'invidus (=frequency) par cluster
nb_cluster = (pd.DataFrame(pred))[0].value_counts()
nb_cluster = pd.DataFrame(nb_cluster)
nb_cluster_sort = nb_cluster.sort_index(axis=0) #tri par index
nb_cluster = nb_cluster.rename(columns={0: 'frequency'})

#compte le nb de cluster pour chaque frequence existante
freq = pd.DataFrame(nb_cluster['frequency'].value_counts())
freq = freq.sort_index(axis=0, ascending=False) #354 clusters composés d'1 seul individu
del nb_cluster, freq

#Cluster summary
cluster_sum = pd.concat([pd.DataFrame(dist_to_next_cluster), nb_cluster_sort], axis=1)
cluster_sum.columns = ['distance to nearest cluster', 'frequency']

# Graph de la distance au cluster le plus proche en fct du nb d'invidus par cluster
plt.plot(cluster_sum['frequency'], cluster_sum['distance to nearest cluster'], 'o')
# y label
plt.ylabel('Distance to nearest cluster')
# x label
plt.xlabel('Frequency of cluster')


#concatene la frequence des clusters à leurs coordonnées des centres
centers2 = pd.concat([pd.DataFrame(centers), pd.DataFrame(nb_cluster_sort)], axis=1)
centers2.columns = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'frequency']
centers2 = centers2[centers2['frequency'] > 10] #633 clusters avec + de 10 individus
centers3 = np.array(centers2.drop('frequency', axis=1))
del nb_cluster_sort, cluster_sum, dist_to_next_cluster, centers, centers2

# Computing euclidian distance of each observations to the nearest cluster
test = vq(data_coor3, centers3)
distances = test[1] # distance de chaque observation au cluster le plus proche.
pylab.ylim([0, 7])
plt.boxplot(distances)
np.percentile(distances, 90)
del test

## Relancer un k-means en virant les observations dont la distance au cluster
# le plus proche est supérieure à 5 et en initialisant les centres précédents
t = pd.DataFrame(distances)
datatest = pd.concat([data_coor3, t], axis=1)
datatest = datatest.rename(columns={0: 'distance'})
datatest_5 = datatest[abs(datatest['distance']) <= 5] #738 individus exclus
del distances, t, datatest


#on fixe le nb de classes au nb de clusters ayant un nb minimal de 10 individus
kmeans = cluster.KMeans(n_clusters=len(centers3), init=centers3)
test = kmeans.fit(pd.DataFrame(np.array(datatest_5)[:, 0:10]))
pred = kmeans.predict(pd.DataFrame(np.array(datatest_5)[:, 0:10]))
nv_centres = test.cluster_centers_
del datatest_5


# K means sur les centres avec 1 iteration sur toutes les observation
kmeans = cluster.KMeans(n_clusters=len(centers3), max_iter=1, init=nv_centres)
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3)
nv_centres_pr_cah = test.cluster_centers_
del nv_centres



########% Clustering hierarchique
# Matrice des distances
Y = pdist(nv_centres_pr_cah, 'euclidean')
hierar = linkage(Y, 'ward')
del Y



## Dendrogramme
plt.figure(figsize=(10, 5))
plt.title('Dendrogramme du clustering hierarchique')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(hierar, color_threshold=35)


## Découpage
groupes_cah = fcluster(hierar, t=4, criterion='maxclust')
plt.hist(groupes_cah)
sum(groupes_cah == 1)
sum(groupes_cah == 2)
sum(groupes_cah == 3)
sum(groupes_cah == 4)
del hierar


# Calcul des centroids de la CAH:
centres_cah = pd.concat([pd.DataFrame(nv_centres_pr_cah), pd.DataFrame(groupes_cah)], axis=1)
centres_cah.columns = ["Comp_" + str(l) for l in list(range(1, 11, 1))] + ['Cluster']
centres_cah_mean = centres_cah.groupby('Cluster').mean()
del centres_cah, nv_centres_pr_cah

## Dernier k-means avec initialisation des centres de la CAH
kmeans = cluster.KMeans(n_clusters=4, max_iter=10000, init=centres_cah_mean)
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3)
del centres_cah_mean
final_centers = test.cluster_centers_


#sauvegarde des clusters obtenus
pd.DataFrame(pred).to_csv(PATH + '/clusters.csv', index=False)
