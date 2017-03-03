# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 12:17:31 2017

@author: Lucie
"""

## Test Gaussian Mixture Models


########## Librairies utilisées
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sn
import sklearn
import sklearn.mixture
#import scipy
#from scipy import linalg
#import matplotlib as mpl



########## Lecture des données


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'


## Import des données
#coordonnees dans les composantes de l'ACP
data_coor2 = pd.read_csv(path + '/PCA_coor2.csv', delimiter=",")
data_coor2 = data_coor2.iloc[:, 0:10]

# Import des données
quanti_trans = pd.read_csv(path + '/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
# Garder que les variables numériques
ech = quanti_trans.drop(['IDPART_CALCULE'], axis=1)

# Echantillonage pour test
echbis = ech.sample(1000, axis=0)

X = data_coor2


### Code plot Bic

print(__doc__)


lowest_bic = np.infty
bic = []
n_components_range = range(1, 10)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(X)
        bic.append(gmm.bic(X))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm

bic = np.array(bic)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue',
                              'darkorange'])
clf = best_gmm
bars = []

# Plot the BIC scores
plt.figure(1, figsize=(8, 6))
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
plt.legend([b[0] for b in bars], cv_types)


## Essai en initialisant les centres

pred = pd.read_csv(path + '/clusters.csv', delimiter=",") +1
quanti_kmeans = pd.concat([ech, pred], axis=1)
quanti_kmeans = quanti_kmeans.rename(columns={'0':'cluster'})
centres_kmeans = quanti_kmeans.groupby('cluster').mean()


test_mod = sklearn.mixture.GaussianMixture(4, covariance_type='diag', means_init=centres_kmeans)
test = test_mod.fit(ech)
test2 = test_mod.predict(ech)
plt.hist(test2)

# Verit box plot


quanti_clus = pd.concat([ech, pd.DataFrame(test2 + 1)], axis=1)
quanti_clus = quanti_clus.rename(columns={0:'cluster'})


# Code pour boxplots
sub = list(range(1, 43, 1))
plt.figure(figsize=(40, 40))
var_names = list(quanti_clus.columns.values)[0:42]
for i in range(0, 42):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=quanti_clus)

