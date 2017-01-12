# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:23:46 2017

@author: Lucie
"""


## Librairies utilisées
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cluster
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



########## Lecture des données


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale/Données'  

# Import des données
quanti_trans = pd.read_csv(path + '/quanti_trans.csv',delimiter=",",dtype={"IDPART_CALCULE":object})
types = quanti_trans.dtypes # Ok
print(types)



########### Boxplot (rapide)

data_boxplot = np.array(quanti_trans)
data_boxplot = data_boxplot[:,1:43]
data_boxplot = data_boxplot.astype(np.float32)
plt.boxplot(data_boxplot)


########## ACP 


# Normaliser les données
base_test = quanti_trans.drop(['IDPART_CALCULE'],axis=1)
data_scale = scale(base_test)



## Code ACP (en utilisant sklearn, on peut utiliser matplotlib qui donne les mêmes résultats)

pca = PCA(n_components=42)
pcafit = pca.fit(data_scale)
var = pca.explained_variance_ratio_

## Graph de la variance expliquée par les composantes
plt.plot(var,'bo')
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.show()

## Graph de la variance cumulée expliquée
plt.plot(np.cumsum(var))
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.show()

## Nouvelle coordonnées
data_coor = pca.transform(data_scale)


## Scatter plot sur les premiers plans de l'ACP
plt.boxplot(data_coor[:,0:40])
plt.scatter(data_coor[:,0],data_coor[:,1])
plt.scatter(data_coor[:,0],data_coor[:,2])
plt.scatter(data_coor[:,1],data_coor[:,2])

## Graph des 3 premieres composantes 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(data_coor[:, 0], data_coor[:, 1], data_coor[:, 2],
cmap=plt.cm.Paired)
ax.set_title("ACP: trois premieres composantes")
ax.set_xlabel("Comp1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Comp2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Comp3")
ax.w_zaxis.set_ticklabels([])



## Concatenation avec les données originales
quanti_acp = pd.concat([quanti_trans,pd.DataFrame(data_coor)],axis=1)





########% Clustering hierarchique
hierar = linkage(data_coor,'ward')

## Dendogramme
plt.figure(figsize=(25,10))
plt.title('Dendogramme du clustering hierarchique')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(hierar,color_threshold=250)


## Decoupage
groupes_cah = fcluster(hierar,t=250,criterion='distance')
plt.hist(groupes_cah)
sum(groupes_cah==1)
sum(groupes_cah==2)
sum(groupes_cah==3)
sum(groupes_cah==4)



## k means
kmeans= cluster.KMeans(n_clusters=4,max_iter =10000)
test = kmeans.fit(pd.DataFrame(data_coor[:,0:10]))
pred = kmeans.predict(pd.DataFrame(data_coor[:,0:10]))


## Comparaison CAH et k-means
pd.crosstab(groupes_cah,pred)


## Analyses des variables sur le premier clustering
quanti_trans['cluster'] = pd.DataFrame(pred)

pd.concat([quanti_trans,pd.DataFrame(pred)])

sn.boxplot(x='cluster', y='Connexion_CAEL_3m', data=quanti_trans)
sn.boxplot(x='cluster', y='Connexion_MaBanque_3m', data=quanti_trans)
sn.boxplot(x='cluster', y='age', data=quanti_trans)
sn.boxplot(x='cluster', y='nb_paiement_chq_3m', data=quanti_trans)
sn.boxplot(x='cluster', y='Agence_3m', data=quanti_trans)
sn.boxplot(x='cluster', y='Consult_Comptes_3m', data=quanti_trans)



