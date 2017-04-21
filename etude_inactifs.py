# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 16:42:38 2017

@author: Lucie
"""



# Etude inactifs

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
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import mixture
import itertools
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.cluster.vq import vq
import sklearn

# Definition du chemin où sont situées les données :
PATH = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'

# Lecture des données
inactifs = pd.read_csv(PATH + '/INACTIFS_CLASSES.csv', delimiter=";",
                 encoding="ISO-8859-1", dtype={"IDPART_CALCULE":object})


inactifs_canaux = pd.read_csv(PATH + '/INACTIFS_CANAUX.csv', delimiter=";",
                 encoding="ISO-8859-1", dtype={"IDPART_CALCULE":object})



### Gaussian mixture model

X = inactifs_canaux.iloc[:,1:29]


test_mod = sklearn.mixture.GaussianMixture(4, covariance_type='diag')
test = test_mod.fit(X)
test2 = test_mod.predict(X)
plt.hist(test2)




# Verif box plot
inactifs_clus = pd.concat([X, pd.DataFrame(test2 + 1)], axis=1)
inactifs_clus = inactifs_clus.rename(columns={0:'cluster'})
# Code pour boxplots
sub = list(range(1, 43, 1))
plt.figure(figsize=(40, 40))
var_names = list(inactifs_clus.columns.values)[0:29]
for i in range(0, 29):
    plt.subplot(6, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=inactifs_clus)
