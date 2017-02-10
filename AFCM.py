# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 10:55:46 2017

@author: Lucie
"""

# AFCM



## Librairies utilisées
import pandas as pd
from mca import mca
import numpy as np
import matplotlib.pyplot as plt
import pylab


## Lecture des données

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données'

# Import des données
base_quali = pd.read_table(path + '/base_variables_quali.txt', delimiter=";", dtype={"IDPART_CALCULE":object})
types = base_quali.dtypes # Ok
print(types)
del(types)


# Import de la base quanti pour recuperation des tops en ligne et dépose
quanti_trans = pd.read_csv(path + '/v2/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
# Concatenation des variables qui nous interessent
base_quali_V2 = pd.concat([base_quali, quanti_trans['nb_contrats_enligne'], quanti_trans['nb_contrats_depose']], axis=1)

# Creation de la variable top
base_quali_V2['top_enligne'] = np.where(base_quali_V2['nb_contrats_enligne'] == 0, 'pas_enligne', 'enligne')
base_quali_V2['top_depose'] = np.where(base_quali_V2['nb_contrats_depose'] == 0, 'pas_depose', 'depose')
base_quali_V2 = base_quali_V2.drop(['nb_contrats_enligne', 'nb_contrats_depose'], axis=1)
base_quali2_V2 = base_quali_V2.drop(['IDPART_CALCULE'], axis=1)
del(base_quali, quanti_trans, path, base_quali_V2)


## Transformation en dummies
dc = pd.DataFrame(pd.get_dummies(base_quali2_V2))

# Selection des 15000 premiers individus (la mca ne tourne pas sur 20000 :-( )
dc_sample = dc.iloc[0:15000, :]
del(dc, base_quali2_V2)

# Code mca
mca_df = mca(dc_sample, benzecri=False)
del dc_sample
coor = mca_df.fs_r(N=10)
plt.boxplot(coor)
coor = pd.DataFrame(coor)

# Enregistrement pour eviter de faire retourner les codes
coor.to_csv(path + '/coor_afcm.csv', index=False)

# Variance expliquée
var_exp = mca_df.L/sum(mca_df.L)
# Variance cumulée
var_cumul = np.cumsum(mca_df.L)/sum(mca_df.L)


## Graph de la variance expliquée par les composantes
plt.title('Var expliquée AFCM')
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var_exp, 'bo')
plt.show()

## Graph de la variance cumulée expliquée
plt.title('Var expliquée cumulée AFCM')
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var_cumul, 'bo')
plt.show()


# eigenvalues
print(mca_df.L)
# Composantes principales des colonnes (modalités)
print(mca_df.fs_c())



# Représentation des modalités dans les plans principaux
# Comp 1 et 2
plt.figure(figsize=(16, 8))
plt.scatter(mca_df.fs_c()[:, 0], mca_df.fs_c()[:, 1])
for i, j, nom in zip(mca_df.fs_c()[:, 0], mca_df.fs_c()[:, 1], dc_sample.columns):
    plt.text(i, j, nom)
plt.show()


# Comp 1 et 3
plt.figure(figsize=(16, 8))
plt.scatter(mca_df.fs_c()[:, 0], mca_df.fs_c()[:, 2])
for i, j, nom in zip(mca_df.fs_c()[:, 0], mca_df.fs_c()[:, 2], dc_sample.columns):
    plt.text(i, j, nom)
plt.show()


# Comp 2 et 3
plt.figure(figsize=(16, 8))
plt.scatter(mca_df.fs_c()[:, 1], mca_df.fs_c()[:, 2])
for i, j, nom in zip(mca_df.fs_c()[:, 1], mca_df.fs_c()[:, 2], dc_sample.columns):
    plt.text(i, j, nom)
plt.show()
