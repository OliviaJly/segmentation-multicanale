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
import pylab
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cluster
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster



########## Lecture des données


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données'  

# Import des données
quanti_trans = pd.read_csv(path + '/quanti_trans.csv',delimiter=",",dtype={"IDPART_CALCULE":object})
types = quanti_trans.dtypes # Ok
print(types)
del(types)
del(path)

########### Boxplot (rapide)

data_boxplot = np.array(quanti_trans)
data_boxplot = data_boxplot[:,1:43]
data_boxplot = data_boxplot.astype(np.float32)

pylab.ylim([-0,20])
plt.boxplot(data_boxplot)
del(data_boxplot)




########## ACP 


# Normaliser les données
base_test = quanti_trans.drop(['IDPART_CALCULE'],axis=1)
data_scale = pd.DataFrame(scale(base_test))
data_scale.columns = [s + '_norm' for s in list(base_test.columns.values)]  # Renomer les colonnes


## Code ACP (en utilisant sklearn, on peut utiliser matplotlib qui donne les mêmes résultats)

pca = PCA(n_components=42)  # setup de l'ACP
pcafit = pca.fit(data_scale) # application sur les données
var = pca.explained_variance_ratio_

## Graph de la variance expliquée par les composantes
pylab.ylim([-0.01,0.3])
pylab.xlim([-1,43])
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var,'bo')
plt.show()

## Graph de la variance cumulée expliquée
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(np.cumsum(var))
plt.show()

## Nouvelle coordonnées
data_coor = pd.DataFrame(pca.transform(data_scale))
data_coor.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes

## Scatter plot sur les premiers plans de l'ACP

pylab.ylim([-7,10])
plt.boxplot(np.array(data_coor)[:,0:40]) # Variance des composantes : décroissante ok

plt.scatter(np.array(data_coor)[:,0],np.array(data_coor)[:,1])
plt.scatter(np.array(data_coor)[:,0],np.array(data_coor)[:,2])
plt.scatter(np.array(data_coor)[:,1],np.array(data_coor)[:,2])

## Graph des 3 premieres composantes 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(np.array(data_coor)[:, 0], np.array(data_coor)[:, 1], np.array(data_coor)[:, 2],
cmap=plt.cm.Paired)
ax.set_title("ACP: trois premieres composantes")
ax.set_xlabel("Comp1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Comp2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Comp3")
ax.w_zaxis.set_ticklabels([])



# 2eme essai ACP en enlevant les variables inintéressantes : 'nb_jours_CAEL_3m',nb_jours_MaBanque_3m,
# Actionsd_MaBanque_3m,Lecture_mess_3m, Ecriture_mess_3m
base_test2 =  base_test.drop(['nb_jours_CAEL_3m','nb_jours_MaBanque_3m','Actionsd_MaBanque_3m',
'Lecture_mess_3m', 'Ecriture_mess_3m'], 1)
data_scale2 = pd.DataFrame(scale(base_test2))
data_scale2.columns = [s + '_norm' for s in list(base_test2.columns.values)]  # Renomer les colonnes




# ACP
pca = PCA(n_components=37)
pcafit = pca.fit(data_scale2)
var = pca.explained_variance_ratio_

## Graph de la variance expliquée par les composantes
pylab.ylim([-0.01,0.3])
pylab.xlim([-1,38])
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var,'bo')
plt.show()

## Graph de la variance cumulée expliquée
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(np.cumsum(var))
plt.show()

## Nouvelle coordonnées
data_coor2 = pd.DataFrame(pca.transform(data_scale2))
data_coor2.columns = ["Comp_" + str(l) for l in list(range(1, 38, 1))] # Renomer les colonnes


## Scatter plot sur les premiers plans de l'ACP

pylab.ylim([-7,10])
plt.boxplot(np.array(data_coor2)[:,0:37])
plt.scatter(np.array(data_coor2)[:,0],np.array(data_coor2)[:,1])
plt.scatter(np.array(data_coor2)[:,0],np.array(data_coor2)[:,2])
plt.scatter(np.array(data_coor2)[:,1],np.array(data_coor2)[:,2])

## Graph des 3 premieres composantes 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(np.array(data_coor2)[:, 0], np.array(data_coor2)[:, 1], np.array(data_coor2)[:, 2],
cmap=plt.cm.Paired)
ax.set_title("ACP: trois premieres composantes")
ax.set_xlabel("Comp1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Comp2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Comp3")
ax.w_zaxis.set_ticklabels([])

## Biplot
xvector = pca.components_[0]
yvector = pca.components_[2]

xs = pca.transform(data_scale2)[:,0]
ys = pca.transform(data_scale2)[:,2]

plt.figure(figsize=(16,8))
plt.title('Représentation des variables dans les composantes 1 et 3')
plt.xlabel('Composante 1')
plt.ylabel('composante 3')

pylab.ylim([-3,2])
pylab.xlim([-3,6])
for i in range(len(xvector)):
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025,zorder=1)
    plt.text(xvector[i]*max(xs)*1.2, yvector[i]*max(ys)*1.2,
             list(base_test2.columns.values)[i], color='r')

for i in range(len(xs)):
# circles project documents (ie rows from csv) as points onto PC axes
    plt.plot(xs[i], ys[i], 'g',zorder=2)
    #plt.text(xs[i]*1.2, ys[i]*1.2, list(base_test2.index)[i], color='b')
plt.show()



## Concatenation avec les données originales
quanti_acp = pd.concat([quanti_trans,data_coor2],axis=1)



########% Clustering hierarchique
hierar = linkage(data_coor2,'ward')

## Dendogramme
plt.figure(figsize=(25,10))
plt.title('Dendogramme du clustering hierarchique')
plt.xlabel('Index')
plt.ylabel('Distance')
dendrogram(hierar,color_threshold=200)


## Découpage
groupes_cah = fcluster(hierar,t=200,criterion='distance')
plt.hist(groupes_cah)
sum(groupes_cah==1)
sum(groupes_cah==2)
sum(groupes_cah==3)
sum(groupes_cah==4)



## k means
kmeans= cluster.KMeans(n_clusters=4,max_iter =10000)
test = kmeans.fit(pd.DataFrame(np.array(data_coor2)[:,0:10]))
pred = kmeans.predict(pd.DataFrame(np.array(data_coor2)[:,0:10]))


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


