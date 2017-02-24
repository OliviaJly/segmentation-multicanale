# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 15:15:59 2017

@author: Lucie
"""

## ACP + Kmeans sur Base mars

# Librairies utilisées
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sn
import pylab
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.vq import vq
from scipy.spatial.distance import pdist
from sklearn import cluster
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import scipy.stats as spstat
from statsmodels.graphics.mosaicplot import mosaic

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/Historique 3 mois'

# Import des données
df_mars = pd.read_csv(path +'/OLIVIA_BASE_QUANTIT_0316.csv',delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE":object, "IDCLI_CALCULE":object})


########## ACP
base_test = df_mars.drop(['IDPART_CALCULE2', 'Actionsd_MaBanque_3m', 'Lecture_mess_3m',
                               'Ecriture_mess_3m'], axis=1)

# Stats moyenne et ecart type
mean = np.mean(base_test, axis=0)
std = np.std(base_test, axis=0)
stats = pd.concat([mean, std], axis=1)
stats.columns = ['mean', 'std']
del mean, std

# Normaliser les données
data_scale = pd.DataFrame(scale(base_test))
data_scale.columns = [s + '_norm' for s in list(base_test.columns.values)]  # Renomer les colonnes


# ACP
pca = PCA(n_components=39)
pcafit = pca.fit(data_scale)
var = pca.explained_variance_ratio_
del var

## Graph de la variance expliquée par les composantes
pylab.ylim([-0.01, 0.3])
pylab.xlim([-1, 40])
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var, 'bo')
plt.show()

## Graph de la variance cumulée expliquée
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(np.cumsum(var))
plt.show()

## Nouvelle coordonnées
score = pca.transform(data_scale)
data_coor = pd.DataFrame(score)
data_coor.columns = ["Comp_" + str(l) for l in list(range(1, 40, 1))] # Renomer les colonnes



## Biplot (Composantes 1 et 2)
vect_propres = pca.components_
xvector = pca.components_[0]#*-1 #1ere composante exprimee dans le referentiel initial des features
yvector = pca.components_[1] #pour inverser l'axe des y, multiplier par -1

xs = score[:, 0] #coordonnees des individus sur 1ere composante
ys = score[:, 1]

plt.figure(figsize=(16, 8))
plt.title('Représentation des variables dans les composantes 1 et 2')
plt.xlabel('Composante 1')
plt.ylabel('composante 2')

pylab.ylim([-4, 3])
pylab.xlim([-3, 5]) #[-4, 4]
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_test.columns.values)[i], color='r')


for i in range(len(xs)): #len(xs) = nb d'invidus
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g', zorder=2)
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(base_test2.index)[i], color='b')
plt.show()


## Biplot (Composantes 1 et 3)
vect_propres = pca.components_
xvector = pca.components_[0]#*-1 #1ere composante exprimee dans le referentiel initial des features
yvector = pca.components_[2] #pour inverser l'axe des y, multiplier par -1

xs = score[:, 0] #coordonnees des individus sur 1ere composante
ys = score[:, 2]

plt.figure(figsize=(16, 8))
plt.title('Représentation des variables dans les composantes 1 et 3')
plt.xlabel('Composante 1')
plt.ylabel('composante 3')

pylab.ylim([-3, 4.5])
pylab.xlim([-3, 5]) #[-4, 4]
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_test.columns.values)[i], color='r')


for i in range(len(xs)): #len(xs) = nb d'invidus
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g', zorder=2)
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(base_test2.index)[i], color='b')
plt.show()




## Biplot (Composantes 2 et 3)
vect_propres = pca.components_
xvector = pca.components_[1]#*-1 #1ere composante exprimee dans le referentiel initial des features
yvector = pca.components_[2] #pour inverser l'axe des y, multiplier par -1

xs = score[:, 1] #coordonnees des individus sur 1ere composante
ys = score[:, 2]

plt.figure(figsize=(16, 8))
plt.title('Représentation des variables dans les composantes 2 et 3')
plt.xlabel('Composante 2')
plt.ylabel('composante 3')

pylab.ylim([-4, 6])
pylab.xlim([-4, 4])
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_test.columns.values)[i], color='r')


for i in range(len(xs)): #len(xs) = nb d'invidus
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g', zorder=2)
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(base_test2.index)[i], color='b')
plt.show()
del i, score, vect_propres, xs, yvector, ys, xvector



#### KMEANS 

# 1er k means
#on retient les 10 premieres composantes
data_coor = data_coor.iloc[:, :10]

kmeans = cluster.KMeans(n_clusters=2000, max_iter=1, n_init=1) #random_state=111
test = kmeans.fit(data_coor)
pred = kmeans.predict(data_coor) #affecte chaque individu au cluster le plus proche

trans = kmeans.transform(data_coor) #donne la distance de chaque individu à chaque cluster
centers = kmeans.cluster_centers_
dist_cluster = kmeans.transform(centers) #matrice de distance entre les clusters
next_cluster = np.argsort(dist_cluster)[:, 1] #cluster le plus proche

dist_to_next_cluster = []      #distance au cluster le plus proche
for i in range(0, len(next_cluster)):
    ind = next_cluster[i]
    dist_to_next_cluster.append(dist_cluster[i, ind])

#compte le nb d'invidus (=frequency) par cluster
nb_cluster = (pd.DataFrame(pred))[0].value_counts()
nb_cluster = pd.DataFrame(nb_cluster)
nb_cluster_sort = nb_cluster.sort_index(axis=0) #tri par index
nb_cluster = nb_cluster.rename(columns={0: 'frequency'})

#compte le nb de cluster pour chaque frequence existante
freq = pd.DataFrame(nb_cluster['frequency'].value_counts())
freq = freq.sort_index(axis=0, ascending=False) #354 clusters composés d'1 seul individu

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


# Computing euclidian distance of each observations to the nearest cluster
test = vq(data_coor, centers3)
distances = test[1] # distance de chaque observation au cluster le plus proche.
pylab.ylim([0, 7])
plt.boxplot(distances)
np.percentile(distances, 90)

## Relancer un k-means en virant les observations dont la distance au cluster
# le plus proche est supérieure à 5 et en initialisant les centres précédents
t = pd.DataFrame(distances)
datatest = pd.concat([data_coor, t], axis=1)
datatest = datatest.rename(columns={0: 'distance'})
datatest_5 = datatest[abs(datatest['distance']) <= 5] #738 individus exclus
del distances, t


#on fixe le nb de classes au nb de clusters ayant un nb minimal de 10 individus
kmeans = cluster.KMeans(n_clusters=len(centers3), init=centers3)
test = kmeans.fit(pd.DataFrame(np.array(datatest_5)[:, 0:10]))
pred = kmeans.predict(pd.DataFrame(np.array(datatest_5)[:, 0:10]))
nv_centres = test.cluster_centers_
# del cent, rand_centroid, i, datatest_5, data_prov


# K means sur les centres avec 1 iteration sur toutes les observation
kmeans = cluster.KMeans(n_clusters=len(centers3), max_iter=1, init=nv_centres)
test = kmeans.fit(data_coor)
pred = kmeans.predict(data_coor)
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
test = kmeans.fit(data_coor)
pred = kmeans.predict(data_coor)
del centres_cah_mean
final_centers = test.cluster_centers_

count=pd.DataFrame(pred+1)[0].value_counts(sort=False)
count2=pd.DataFrame(count)
del centers, centers2, centers3, cluster_sum, count, datatest, datatest_5, dist_cluster, dist_to_next_cluster, freq, groupes_cah,
del i, nb_cluster, next_cluster, trans, ind, nb_cluster_sort

# Bar plot freq cluster
plt.figure(figsize=(12, 8))
ax = count.plot(kind='bar')
ax.set_title("Nb d'individus par classe")
ax.set_xlabel("Classe")
ax.set_ylabel("Nb d'individus")
rects = ax.patches
labels = [count2.iat[i,0] for i in range(len(rects))] 
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
del labels, height, count2
# Concatenation avec les variables initiales
clustered_data = pd.concat([base_test, pd.DataFrame(pred)], axis=1)
clustered_data = clustered_data.rename(columns={0: 'cluster'}) # Attention au type du nom de cluster
clustered_data['cluster']=clustered_data['cluster']+1



# Premières analyses
analyses_kmeans = clustered_data.groupby('cluster').mean()
# Code pour boxplots
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(clustered_data.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=clustered_data)
del sub, var_names

# Base acp avec les clusters
C = np.array(pd.concat([data_coor, clustered_data['cluster']], axis=1))
# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 (Inactifs)')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 (Retraites)')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 (CAEL)')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 (Ma Banque)')
ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n MA Banque -- CAEL")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n  Jeunes -- Agés")
#ax.w_zaxis.set_ticklabels([])
plt.legend()
del label, C


# Import des bases de Juin et Septembre pour comparaison
# Juin
df_juin = pd.read_csv(path +'/OLIVIA_BASE_QUANTIT_0616.csv',delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE":object, "IDCLI_CALCULE":object})
base_juin = df_juin.drop(['IDPART_CALCULE2', 'Actionsd_MaBanque_3m', 'Lecture_mess_3m',
                               'Ecriture_mess_3m'], axis=1)
data_scale_juin = pd.DataFrame(scale(base_juin))
data_scale_juin.columns = [s + '_norm' for s in list(base_juin.columns.values)]  # Renomer les colonnes
## Nouvelle coordonnées juin
score_juin = pca.transform(data_scale_juin)
data_coor_juin = pd.DataFrame(score_juin)
data_coor_juin.columns = ["Comp_" + str(l) for l in list(range(1, 40, 1))] # Renomer les colonnes
data_coor_juin = data_coor_juin.iloc[:,0:10]
# Reatribution des cluster :
nearest_clus = vq(data_coor_juin, final_centers)[0]+1
# Recup var initiales
base_juin = pd.concat([base_juin,pd.DataFrame(nearest_clus)],axis=1)
base_juin = base_juin.rename(columns={0: 'cluster'})
count = base_juin['cluster'].value_counts()
# Analyses
analyses_kmeans_juin = base_juin.groupby('cluster').mean()
# Code pour boxplots
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(base_juin.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=base_juin)
del sub, var_names

# Base acp avec les clusters
C = np.array(pd.concat([data_coor_juin,base_juin['cluster']],axis=1))
# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 (Inactifs)')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 (Retraites)')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 (CAEL)')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 (Ma Banque)')
ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n MA Banque -- CAEL")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n  Jeunes -- Agés")
#ax.w_zaxis.set_ticklabels([])
plt.legend()


# Septembre
df_sept = pd.read_csv(path +'/OLIVIA_BASE_QUANTIT_0916.csv',delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE":object, "IDCLI_CALCULE":object})
base_sept = df_sept.drop(['IDPART_CALCULE2', 'Actionsd_MaBanque_3m', 'Lecture_mess_3m',
                               'Ecriture_mess_3m'], axis=1)
data_scale_sept = pd.DataFrame(scale(base_sept))
data_scale_sept.columns = [s + '_norm' for s in list(base_sept.columns.values)]  # Renomer les colonnes
## Nouvelle coordonnées juin
score_sept = pca.transform(data_scale_sept)
data_coor_sept = pd.DataFrame(score_sept)
data_coor_sept.columns = ["Comp_" + str(l) for l in list(range(1, 40, 1))] # Renomer les colonnes
data_coor_sept = data_coor_sept.iloc[:,0:10]
# Reatribution des cluster :
nearest_clus_sept = vq(data_coor_sept, final_centers)[0]+1
# Recup var initiales
base_sept = pd.concat([base_sept,pd.DataFrame(nearest_clus_sept)],axis=1)
base_sept = base_sept.rename(columns={0: 'cluster'})
count = base_sept['cluster'].value_counts()
# Analyses
analyses_kmeans_sept = base_sept.groupby('cluster').mean()
# Code pour boxplots
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(base_sept.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=base_sept)
del sub, var_names

# Base acp avec les clusters
C = np.array(pd.concat([data_coor_sept,base_sept['cluster']],axis=1))
# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 (Inactifs)')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 (Retraites)')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 (CAEL)')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 (Ma Banque)')
ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n MA Banque -- CAEL")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n  Jeunes -- Agés")
#ax.w_zaxis.set_ticklabels([])
plt.legend()


# Matrices de transition

mat_transition = pd.concat([clustered_data['cluster'],base_juin['cluster'],base_sept['cluster']],axis=1)
mat_transition.columns = ['Mars','Juin','Septembre']

# Cross tab mars juin
pd.crosstab(mat_transition['Mars'],mat_transition['Juin'])
pd.crosstab(mat_transition['Mars'],mat_transition['Juin']).apply(lambda r: r/r.sum(), axis=1)

# Cross tab mars septembre
pd.crosstab(mat_transition['Mars'],mat_transition['Septembre'])
pd.crosstab(mat_transition['Mars'],mat_transition['Septembre']).apply(lambda r: r/r.sum(), axis=1)

# Cross tab juin septembre
pd.crosstab(mat_transition['Juin'],mat_transition['Septembre'])
pd.crosstab(mat_transition['Juin'],mat_transition['Septembre']).apply(lambda r: r/r.sum(), axis=1)


mat_transition['Septembre'].value_counts(sort=False)
mat_transition['Mars'].value_counts(sort=False)
mat_transition['Juin'].value_counts(sort=False)

test =  mat_transition[mat_transition.Mars==mat_transition.Septembre]
test =  test[test.Mars==test.Juin]
# 16466 n'ont pas changé de classes


sum(mat_transition['Mars'] == mat_transition['Juin'] & mat_transition['Juin'] == mat_transition['Septembre'])
a = np.array(mat_transition['Mars'] == mat_transition['Juin'])
b = np.array(mat_transition['Juin'] == mat_transition['Septembre'])
c = (a) and (b)
