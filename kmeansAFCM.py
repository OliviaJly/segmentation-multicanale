# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 15:10:50 2017

@author: Lucie
"""


########## K-means et CAH SUR COOR AFCM - Même méthode que SAS


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


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données'

# Import des données coor AFCM
data_coor_afcm = pd.read_csv(path + '/coor_afcm.csv', delimiter=",")

# Import des données quali
base_quali = pd.read_table(path + '/base_variables_quali.txt',delimiter=";",dtype={"IDPART_CALCULE":object})




# Import de la base quanti pour recuperation des tops en ligne et dépose
quanti_trans = pd.read_csv(path + '/v2/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
base_test2 = quanti_trans.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', \
                                'Lecture_mess_3m', 'Ecriture_mess_3m'], 1)
base_test2 = base_test2.iloc[0:15000,:]

base_quali2 = pd.concat([base_quali,quanti_trans['nb_contrats_enligne'],quanti_trans['nb_contrats_depose']],axis=1)
base_quali2['top_enligne'] = np.where(base_quali2['nb_contrats_enligne'] == 0, 'pas_enligne', 'enligne')
base_quali2['top_depose'] = np.where(base_quali2['nb_contrats_depose'] == 0, 'pas_depose', 'depose')
base_quali2 = base_quali2.drop(['nb_contrats_enligne','nb_contrats_depose'],axis=1)
base_quali2 = base_quali2.iloc[0:15000,:]
del(base_quali,quanti_trans,path)




# 1er k means

kmeans = cluster.KMeans(n_clusters=1500, max_iter=1, n_init=1) #random_state=111
test = kmeans.fit(data_coor_afcm)
pred = kmeans.predict(data_coor_afcm) #affecte chaque individu au cluster le plus proche

trans = kmeans.transform(data_coor_afcm) #donne la distance de chaque individu à chaque cluster
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
freq = freq.sort_index(axis=0, ascending=False)

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
centers2 = centers2[centers2['frequency'] > 10] 
centers3 = np.array(centers2.drop('frequency', axis=1))


# Computing euclidian distance of each observations to the nearest cluster
test = vq(data_coor_afcm, centers3)
distances = test[1] # distance de chaque observation au cluster le plus proche.
pylab.ylim([0, 0.8])
plt.boxplot(distances)
np.percentile(distances, 90)

## Relancer un k-means en virant les observations dont la distance au cluster
# le plus proche est supérieure à 5 et en initialisant les centres précédents
t = pd.DataFrame(distances)
datatest = pd.concat([data_coor_afcm, t], axis=1)
datatest = datatest.rename(columns={0: 'distance'})
#datatest_5 = datatest[abs(datatest['distance']) <= 5] #738 individus exclus 
# Ici par besoin d'enlever les individus trop distants car moins de variabilité : la distribution
# des distances est moins étalée
del distances, t


#on fixe le nb de classes au nb de clusters ayant un nb minimal de 10 individus
kmeans = cluster.KMeans(n_clusters=len(centers3), init=centers3)
test = kmeans.fit(pd.DataFrame(np.array(datatest)[:, 0:10]))
pred = kmeans.predict(pd.DataFrame(np.array(datatest)[:, 0:10]))
nv_centres = test.cluster_centers_
# del cent, rand_centroid, i, datatest_5, data_prov


# K means sur les centres avec 1 iteration sur toutes les observation
kmeans = cluster.KMeans(n_clusters=len(centers3), max_iter=1, init=nv_centres)
test = kmeans.fit(pd.DataFrame(np.array(data_coor_afcm)))
pred = kmeans.predict(pd.DataFrame(np.array(data_coor_afcm)))
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
dendrogram(hierar, color_threshold=5)


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
test = kmeans.fit(pd.DataFrame(np.array(data_coor_afcm)))
pred = kmeans.predict(pd.DataFrame(np.array(data_coor_afcm)))
plt.hist(pred)
del centres_cah_mean

# Recuperation des variables initiales
clustered_data = pd.concat([base_test2, pd.DataFrame(pred)], axis=1)
clustered_data = clustered_data.rename(columns={0: 'cluster'})
sum(pred == 0)
sum(pred == 1)
sum(pred == 2)
sum(pred == 3)



# Premières analyses
analyses_kmeans = clustered_data.groupby('cluster').mean()

# Code pour boxplots
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(clustered_data.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=clustered_data)




## Représentation dans le plan de l'AFCM des clusters

# Base acp avec les clusters
C = np.array(pd.concat([pd.DataFrame(np.array(data_coor_afcm)), \
                                     clustered_data['cluster']], axis=1))


# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 0, 0], C[C[:, 10] == 0, 1], C[C[:, 10] == 0, 2], \
           c='royalblue', cmap=plt.cm.Paired, label='Groupe 1')
ax.scatter(C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], C[C[:, 10] == 1, 2], \
           c='forestgreen', cmap=plt.cm.Paired, label='Groupe 2')
ax.scatter(C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], C[C[:, 10] == 2, 2], \
           c='firebrick', cmap=plt.cm.Paired, label='Groupe 3')
ax.scatter(C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], C[C[:, 10] == 3, 2], \
           c='slateblue', cmap=plt.cm.Paired, label='Groupe 4')
ax.set_title("Représentation des partenaires \n dans la segmentation")
ax.set_xlabel("\n Composante 1")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 2")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Composante 3")
#ax.w_zaxis.set_ticklabels([])
plt.legend()
plt.savefig('Plot_3_comp_AFCM.png', dpi=600)







# Essai plot 3D interactif
import plotly
import plotly.graph_objs as go
import numpy as np
from plotly.graph_objs import *

plotly.tools.set_credentials_file(username='luciemallet', api_key='EchUlInyh3yStYnkmbhD')

x1, y1, z1 =C[C[:, 10] == 0, 0], C[C[:, 10] == 0, 1], C[C[:, 10] == 0, 2]
trace0 = go.Scatter3d(
    x=x1,
    y=y1,
    z=z1,
    mode='markers',
    marker=dict(
        color='blue',
        size=5,
        symbol='circle',
        line=dict(
            color='blue',
            width=1
        ),
        opacity=0.8
    )
)

x2, y2, z2 =C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], C[C[:, 10] == 1, 2]
trace1 = go.Scatter3d(
    x=x2,
    y=y2,
    z=z2,
    mode='markers',
    marker=dict(
        color='green',
        size=5,
        symbol='diamond',
        line=dict(
            color='green',
            width=1
        ),
        opacity=0.8
    )
)
        
        
x3, y3, z3 = C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], C[C[:, 10] == 2, 2]
trace2 = go.Scatter3d(
    x=x3,
    y=y3,
    z=z3,
    mode='markers',
    marker=dict(
        color='red',
        size=5,
        symbol='cross',
        line=dict(
            color='red',
            width=1
        ),
        opacity=0.8
    )
)
        
x4, y4, z4 = C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], C[C[:, 10] == 3, 2]
trace3 = go.Scatter3d(
    x=x4,
    y=y4,
    z=y4,
    mode='markers',
    marker=dict(
        color='purple',
        size=5,
        symbol='triangle-up',
        line=dict(
            color='purple',
            width=1
        ),
        opacity=0.8
    )
)
        
data = [trace0, trace1, trace2, trace3]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene=Scene(
        xaxis=XAxis(title='x axis title'),
        yaxis=YAxis(title='y axis title'),
        zaxis=ZAxis(title='z axis title')
    )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='repr_cluster_AFCM')





# Creation des tops en ligne et depose
clustered_data['top_enligne'] = np.where(clustered_data['nb_contrats_enligne'] == 0, 0, 1)
clustered_data['top_depose'] = np.where(clustered_data['nb_contrats_depose'] == 0, 0, 1)

# Bar plot des tops par cluster
# Moyenne des tops par cluster
mean_top_depose = clustered_data.groupby('cluster')['top_depose'].mean()
mean_top_enligne = clustered_data.groupby('cluster')['top_enligne'].mean()

barWidth = 0.4
y1 = mean_top_depose
y2 = mean_top_enligne
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, y1, width=barWidth, color=['yellow' for i in y1], label='Top depose')
plt.bar(r2, y2, width=barWidth, color=['pink' for i in y1], label='Top en ligne')
plt.xticks([r + barWidth for r in range(len(y1))], ['Cluster 1 (Inactifs)', \
'Cluster 2 (Retraités)', 'Cluster 3 (CAEL)', 'Cluster 4 (MB)'])
plt.suptitle('Nb moyen de Top en ligne et Top depose par cluster')
plt.legend()
plt.savefig('TOP_par_clusters', dpi=600)






# En résumé :

# Pourcentage de TOP dans toute la base
(sum(clustered_data['top_enligne']) / len(clustered_data.axes[0])) * 100 # 0.4 %
(sum(clustered_data['top_depose']) / len(clustered_data.axes[0])) * 100 # 2.96 %

# Pourcentage dans les clusters
mean_top_enligne * 100
mean_top_depose * 100 




# Ajout de test du CHI2
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(clustered_data['top_enligne'],clustered_data['cluster']))
p # Reject H0
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(clustered_data['top_depose'],clustered_data['cluster']))
p # Reject H0
