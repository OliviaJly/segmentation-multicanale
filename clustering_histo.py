# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 15:57:11 2017

clustering sur var quanti (ACP + k means) sur les 20 000 individus sur tout l'historique (11 mois)

@author: Richard
"""

# Librairies utilisées
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cluster
import seaborn as sn

########### IMPORT DES DONNEES :

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/Historique 3 mois'

# Import des données
base_quanti = pd.read_csv(path +'/OLIVIA_BASE_QUANTI_SEL2.csv', nrows=200000, delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE2":object}) #var non transformees

base_quantit = pd.read_csv(path +'/OLIVIA_BASE_QUANTIT2.csv', nrows=200000, delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE2":object}) #var transformees


########## ACP
base_acp = base_quantit.drop(['IDPART_CALCULE2','date_part'], axis=1)

# Stats moyenne et ecart type
mean = np.mean(base_acp, axis=0)
std = np.std(base_acp, axis=0)
stats = pd.concat([mean, std], axis=1)
stats.columns = ['mean', 'std']
del(mean, std)

# Normaliser les données
data_scale = pd.DataFrame(scale(base_acp))
data_scale.columns = [s + '_norm' for s in list(base_acp.columns.values)]  # Renomer les colonnes

# ACP
pca = PCA(n_components=42)
pcafit = pca.fit(data_scale)
var = pca.explained_variance_ratio_
del var
## Nouvelles coordonnées
score = pca.transform(data_scale)
data_coor = pd.DataFrame(score)
data_coor.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes


## Graph de la variance expliquée par les composantes
pylab.ylim([-0.01, 0.3])
pylab.xlim([-1, 40])
plt.ylabel('Part de la variance expliquée')
plt.xlabel('Composantes')
plt.plot(var, 'bo')
plt.show()

## Graph de la variance cumulée expliquée
plt.ylabel('Part de la variance expliquée cumulée')
plt.xlabel('Composantes')
plt.plot(np.cumsum(var))
plt.show()


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

pylab.ylim([-5, 4])
pylab.xlim([-3.5, 6.5]) #[-4, 4]
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_acp.columns.values)[i], color='r')

plt.savefig('Biplot 1 et 2.png', dpi=600) 
plt.show()

## Biplot (Composantes 1 et 3)
xvector = pca.components_[0]#*-1 #1ere composante exprimee dans le referentiel initial des features
yvector = pca.components_[2] #pour inverser l'axe des y, multiplier par -1

xs = score[:, 0] #coordonnees des individus sur 1ere composante
ys = score[:, 2]

plt.figure(figsize=(16, 8))
plt.title('Représentation des variables dans les composantes 1 et 3')
plt.xlabel('Composante 1')
plt.ylabel('composante 3')

pylab.ylim([-3, 2])
pylab.xlim([-3.5, 6.5]) #[-4, 4]
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_acp.columns.values)[i], color='r')
    
plt.savefig('Biplot 1 et 3.png', dpi=600) 
plt.show()

## Biplot (Composantes 2 et 3)
xvector = pca.components_[1]#*-1 #1ere composante exprimee dans le referentiel initial des features
yvector = pca.components_[2] #pour inverser l'axe des y, multiplier par -1

xs = score[:, 1] #coordonnees des individus sur 1ere composante
ys = score[:, 2]

plt.figure(figsize=(16, 8))
plt.title('Représentation des variables dans les composantes 2 et 3')
plt.xlabel('Composante 2')
plt.ylabel('composante 3')

pylab.ylim([-3, 2])
pylab.xlim([-5, 4.5])
for i in range(len(xvector)): #len(xvector) = nb features
# arrows project features (ie columns from csv) as vectors onto PC axes
    plt.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
              color='r', width=0.0005, head_width=0.0025, zorder=1)
    plt.text(xvector[i]*max(xs)*1.1, yvector[i]*max(ys)*1.1,
             list(base_acp.columns.values)[i], color='r')
    
plt.savefig('Biplot 2 et 3.png', dpi=600) 
plt.show()

###### K MEANS 
# 1er k means
# on retient les 10 premieres composantes
data_coor2 = data_coor.iloc[:, :10]
test = -np.array(data_coor2['Comp_3'])
data_coor3 = pd.concat([data_coor2.iloc[:,0:2],pd.DataFrame(test),data_coor2.iloc[:,3:10]],axis=1)
data_coor3 = data_coor3.rename(columns={0: 'Comp_3'})

kmeans = cluster.KMeans(n_clusters=22000, max_iter=1, n_init=1) #random_state=111
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3) #affecte chaque individu au cluster le plus proche

#### semble long à s'exécuter ... 
#### suite du clustering sous SAS - clustering dynamique 


data_coor3.to_csv(path + '/coord_ACP_histo.csv', index=False)



#evol des variables sur tout l'historique
#Attention ne pas executer  
base_quanti["date_part"] = pd.to_datetime(base_quanti["date_part"])

sub = list(range(1, 42, 1))
plt.figure(figsize=(42, 42))
var_names = list(base_acp.columns.values)[0:42]
for i in range(0, 41):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='date_part', y=var_names[i], data=base_quanti)

# var transformées
base_quantit["date_part"] = pd.to_datetime(base_quantit["date_part"])

sub = list(range(1, 43, 1))
plt.figure(figsize=(43, 43))
var_names = list(base_acp.columns.values)[0:42]
for i in range(0, 42):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='date_part', y=var_names[i], data=base_quantit)
##fin 
    
    
# boxplots individuels pour présenter certains comportements
sn.boxplot(x='date_part',y='Connexion_MaBanque_3m', data=base_quantit)
sn.boxplot(x='date_part',y='Connexion_CAEL_3m', data=base_quantit)

#calcul de moyennes sur var transf
mean_var = base_quantit.groupby('date_part').mean()
med_var = base_quantit.groupby('date_part').median()

plt.plot(mean_var['Connexion_MaBanque_3m'])
plt.plot(mean_var['Connexion_CAEL_3m'])

#calcul de moyennes sur var non transf
mean_var_nt = base_quanti.groupby('date_part').mean()
mean_var_nt = base_quanti.groupby('date_part').mean()


plt.plot(mean_var_nt['Connexion_MaBanque_3m'])
plt.plot(mean_var_nt['Connexion_CAEL_3m'])
plt.plot(mean_var_nt['age'])
plt.plot(mean_var_nt['nb_paiement_chq_3m'])
plt.plot(mean_var_nt['nb_paiement_carte_3m'])
plt.plot(mean_var_nt['REVENU_EST_MM'])
plt.plot(mean_var_nt['SURFACE_FINANCIERE'])
plt.plot(mean_var_nt['ENCOURS_DAV'])
plt.plot(mean_var_nt['Automate_3m'])
plt.plot(mean_var_nt['Agence_3m'])


#calcul de moyennes sur var non transf connexion >0
connexion_sup0_MB =  base_quanti[base_quanti.Connexion_MaBanque_3m>0]
connexion_sup0_CAEL =  base_quanti[base_quanti.Connexion_CAEL_3m>0]

mean_var_nt_0_MB = connexion_sup0_MB.groupby('date_part').mean()
mean_var_nt_0_CAEL = connexion_sup0_CAEL.groupby('date_part').mean()

plt.plot(mean_var_nt_0_MB['Connexion_MaBanque_3m'])
plt.plot(mean_var_nt_0_CAEL['Connexion_CAEL_3m'])
