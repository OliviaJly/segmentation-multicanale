# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:23:46 2017

@author: Lucie
"""


########## Librairies utilisées
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D


########## Lecture des données


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'

# Import des données
quanti_trans = pd.read_csv(path + '/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
types = quanti_trans.dtypes # Ok
print(types)
del types


########### Boxplot (rapide)
data_boxplot = np.array(quanti_trans)
data_boxplot = data_boxplot[:, 1:43]
data_boxplot = data_boxplot.astype(np.float32)

pylab.ylim([-0, 20])
plt.boxplot(data_boxplot)
del data_boxplot



########## ACP
base_test = quanti_trans.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', 'Lecture_mess_3m',
                               'Ecriture_mess_3m'], axis=1)

# Stats moyenne et ecart type
mean = np.mean(base_test, axis=0)
std = np.std(base_test, axis=0)
stats = pd.concat([mean, std], axis=1)
stats.columns = ['mean', 'std']
del mean, std, stats

# Normaliser les données
data_scale = pd.DataFrame(scale(base_test))
data_scale.columns = [s + '_norm' for s in list(base_test.columns.values)]  # Renomer les colonnes


# ACP
pca = PCA(n_components=39)
pcafit = pca.fit(data_scale)
var = pca.explained_variance_ratio_


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
del var

## Nouvelle coordonnées
score = pca.transform(data_scale)
data_coor = pd.DataFrame(score)
data_coor.columns = ["Comp_" + str(l) for l in list(range(1, 40, 1))] # Renomer les colonnes
del data_scale

## Enregistrement des données data_coor pour programme kmeans_outliers
data_coor.to_csv(path + '/PCA_coor2.csv', index=False)


## Scatter plot des individus sur les premiers plans de l'ACP

pylab.ylim([-7, 10])
plt.boxplot(np.array(data_coor)[:, 0:39])
plt.scatter(np.array(data_coor)[:, 0], np.array(data_coor)[:, 1])
plt.scatter(np.array(data_coor)[:, 0], np.array(data_coor)[:, 2])
plt.scatter(np.array(data_coor)[:, 1], np.array(data_coor)[:, 2])

## Graph des 3 premieres composantes 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(np.array(data_coor)[:, 0], np.array(data_coor)[:, 1], \
           np.array(data_coor)[:, 2], cmap=plt.cm.Paired)
ax.set_title("ACP: trois premieres composantes")
ax.set_xlabel("Comp1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Comp2")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Comp3")
ax.w_zaxis.set_ticklabels([])



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


del i, score, vect_propres, xs, xvector, ys, yvector
