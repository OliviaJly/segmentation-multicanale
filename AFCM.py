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


## Lecture des données

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données'  

# Import des données
base_quali = pd.read_table(path + '/base_variables_quali.txt',delimiter=";",dtype={"IDPART_CALCULE":object})
types = base_quali.dtypes # Ok
print(types)
del(types)



## AFCM

base_quali2 = base_quali.drop(['IDPART_CALCULE'],axis=1)
dc=pd.DataFrame(pd.get_dummies(base_quali2))
dc.head()
del(base_quali)
del(base_quali2)

mca_df=mca(dc,benzecri=False)
# Valeurs singulières
print(mca_df.L)
# Composantes principales des colonnes (modalités)
print(mca_df.fs_c())
# Premier plan principal
plt.figure(figsize=(26,13))
plt.scatter(mca_df.fs_c()[:, 0],mca_df.fs_c()[:, 1])
for i, j, nom in zip(mca_df.fs_c()[:, 0],mca_df.fs_c()[:, 1], dc.columns):
  plt.text(i, j, nom)
plt.show()
