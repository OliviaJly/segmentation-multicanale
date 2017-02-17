# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 15:38:37 2017

@author: Lucie
"""


########## K-means et CAH - Même méthode que SAS


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

# Definition du chemin où sont situées les données :
PATH = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'

## Import des données
#coordonnees dans les composantes de l'ACP
data_coor2 = pd.read_csv(PATH + '/PCA_coor2.csv', delimiter=",") 

#base quanti ayant servi pour l'ACP
quanti_trans = pd.read_csv(PATH + '/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
base_test2 = quanti_trans.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', \
                                'Lecture_mess_3m', 'Ecriture_mess_3m'], 1) 

#base quanti avant transformation des variables
base_quanti = pd.read_csv(PATH + '/base_quanti.csv', delimiter=",", dtype={"IDPART_CALCULE":object})

del quanti_trans

# 1er k means
#on retient les 10 premieres composantes
data_coor3 = data_coor2.iloc[:, :10]

kmeans = cluster.KMeans(n_clusters=2000, max_iter=1, n_init=1) #random_state=111
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3) #affecte chaque individu au cluster le plus proche

trans = kmeans.transform(data_coor3) #donne la distance de chaque individu à chaque cluster
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
test = vq(data_coor3, centers3)
distances = test[1] # distance de chaque observation au cluster le plus proche.
pylab.ylim([0, 7])
plt.boxplot(distances)
np.percentile(distances, 90)

## Relancer un k-means en virant les observations dont la distance au cluster
# le plus proche est supérieure à 5 et en initialisant les centres précédents
t = pd.DataFrame(distances)
datatest = pd.concat([data_coor3, t], axis=1)
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
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3)
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
test = kmeans.fit(data_coor3)
pred = kmeans.predict(data_coor3)
del centres_cah_mean
final_centers = test.cluster_centers_


#sauvegarde des clusters obtenus 
pd.DataFrame(pred).to_csv(PATH + '/clusters.csv', index=False)


pred = pd.read_csv(PATH + '/clusters.csv', delimiter=",") 



# frequence des clusters
count=pd.DataFrame(pred+1)[0].value_counts(sort=False)
count2=pd.DataFrame(count)

# now to plot the figure...
plt.figure(figsize=(12, 8))
ax = count.plot(kind='bar')
ax.set_title("Nb d'individus par classe")
ax.set_xlabel("Classe")
ax.set_ylabel("Nb d'individus")
rects = ax.patches
# Now make some labels
labels = [count2.iat[i,0] for i in range(len(rects))] 
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.savefig('frequence clusters.png', dpi=600) 


# Recuperation des variables initiales
clustered_data = pd.concat([base_test2, pd.DataFrame(pred)], axis=1)
clustered_data = clustered_data.rename(columns={0: 'cluster'}) # Attention au type du nom de cluster
clustered_data['cluster']=clustered_data['cluster']+1



clustered_data2 = pd.concat([base_quanti, pd.DataFrame(pred)], axis=1)
clustered_data2 = clustered_data2.rename(columns={'0': 'cluster'})
clustered_data2['cluster']=clustered_data2['cluster']+1

####### Ajout comparaison historique
base_juin = pd.read_csv('C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/Historique 3 mois/quanti_trans2juin.csv', delimiter=",") 
base_juin = base_juin.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', 'Lecture_mess_3m',
                               'Ecriture_mess_3m'], axis=1)
# Centres quanti
centres_quanti = clustered_data.groupby('cluster').mean()
# Pred Juin : 
kmeans = cluster.KMeans(n_clusters=4, max_iter=1, init=centres_quanti)
test = kmeans.fit(base_juin)
pred_juin = kmeans.predict(base_juin)

# Comparaison des prediction
pred_compare = pd.concat([pd.DataFrame(pred),pd.DataFrame(pred_juin)],axis=1)

pred_compare = pd.DataFrame({'sept': pred, 'juin': pred_juin})
pd.crosstab(pred_compare['sept']+1,pred_compare['juin']+1)

base_juin2 = pd.concat([base_juin, pd.DataFrame(pred_juin)],axis= 1)
base_juin2 = base_juin.rename(columns={0: 'cluster'})

test_dist = vq(base_juin, centres_quanti)
plus_proche = test_dist[0] # distance de chaque observation au cluster le plus proche.

pred_compare2 = pd.concat([pred_compare,pd.DataFrame(plus_proche)],axis=1)
pred_compare2 = pred_compare2.rename(columns={0: 'juindistance'})
pd.crosstab(pred_compare2['juin']+1,pred_compare2['juindistance']+1)
pd.crosstab(pred_compare2['sept']+1,pred_compare2['juindistance']+1)


# Boxplot nouvelle pred
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(base_juin2.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=base_juin2)


    
    

    
    

# Premières analyses
analyses_kmeans = clustered_data.groupby('cluster').mean()
analyses_kmeans2 = clustered_data2.groupby('cluster').mean()

# Code pour boxplots
sub = list(range(1, 40, 1))
plt.figure(figsize=(40, 40))
var_names = list(clustered_data.columns.values)[0:39]
for i in range(0, 39):
    plt.subplot(8, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=clustered_data)

# boxplots individuels pour présenter certains comportements
sn.boxplot(x='cluster',y='age', data=clustered_data)
sn.plt.suptitle("Distribution de l'age par classe")   
plt.savefig('distrib age par classe.png', dpi=600) 

sn.boxplot(x='cluster',y='nb_mois_dern_entr', data=clustered_data)
sn.plt.suptitle("Nb de mois depuis le dernier entretien par classe")    
plt.savefig('distrib nb_mois_dern_entr par classe.png', dpi=600) 

#les familles (avec nb_partenaires >= 3) se retrouvent plutot dans les cluster 1 et 3
#en effet, les inactifs et retraites sont plutot seuls ou à 2 (0 mineurs dans le CC)  
sn.boxplot(x='cluster',y='nb_partenaires_CC', data=clustered_data)
sn.boxplot(x='cluster',y='nb_mineurs_CC', data=clustered_data)

pd.crosstab(clustered_data['cluster'],clustered_data['nb_partenaires_CC'])
pd.crosstab(clustered_data['cluster'],clustered_data['nb_mineurs_CC'])

# Connexion CAEL
#avec les donnees transformees
sn.boxplot(x='cluster',y='Connexion_CAEL_3m', data=clustered_data)
sn.plt.suptitle("Nb de connexions à CAEL sur les 3 derniers mois par classe")    

#avec les donnees initiales 
ax=sn.boxplot(x='cluster',y='Connexion_CAEL_3m', data=clustered_data2, showmeans=True)
ax.set(ylim=(0, 100))
sn.plt.suptitle("Nb de connexions à CAEL sur les 3 derniers mois par classe")    
plt.savefig('distrib Connexion_CAEL par classe.png', dpi=600) 

# Connexion MaBanque 
#avec les donnees transformees
sn.boxplot(x='cluster',y='Connexion_MaBanque_3m', data=clustered_data)
sn.plt.suptitle("Nb de connexions à Ma Banque sur les 3 derniers mois par classe")    

#avec les donnees initiales 
ax=sn.boxplot(x='cluster',y='Connexion_MaBanque_3m', data=clustered_data2, showmeans=True)
ax.set(ylim=(0, 65))
sn.plt.suptitle("Nb de connexions à Ma Banque sur les 3 derniers mois par classe")
plt.savefig('distrib Connexion_MaBanque par classe.png', dpi=600) 

# nb paiement par carte -> la classe des retraites en effectue le -
sn.boxplot(x='cluster',y='nb_paiement_carte_3m', data=clustered_data)
sn.plt.suptitle("Nb de paiements par carte sur les 3 derniers mois par classe")  
plt.savefig('distrib nb_paiement_carte par classe.png', dpi=600)   

# Montant operation depot pour montrer le peu d'activite sur le compte de la classe des "inactifs" 
#avec les donnees transformees
sn.boxplot(x='cluster',y='MT_OPERATION_DEPOT_3m', data=clustered_data)
sn.plt.suptitle("Mt des operations de depot sur les 3 derniers mois par classe")  

#avec les donnees initiales 
ax=sn.boxplot(x='cluster',y='MT_OPERATION_DEPOT_3m', data=clustered_data2, showmeans=True)
ax.set(ylim=(0, 60000))
sn.plt.suptitle("Mt des operations de depot sur les 3 derniers mois par classe")  
plt.savefig('distrib mt_operation_depot par classe.png', dpi=600)     

# Agence : le seul contact que possede la classe des retraites
#avec les donnees transformees
sn.boxplot(x='cluster',y='Agence_3m', data=clustered_data)
sn.plt.suptitle("Nb de contacts agence sur les 3 derniers mois par classe")  

#avec les donnees initiales 
ax=sn.boxplot(x='cluster',y='Agence_3m', data=clustered_data2, showmeans=True)
ax.set(ylim=(0, 20))
sn.plt.suptitle("Nb de contacts agence sur les 3 derniers mois par classe")  
plt.savefig('distrib agence par classe.png', dpi=600)     






## Représentation dans le plan de l'ACP des clusters

# Base acp avec les clusters
C = np.array(pd.concat([data_coor3, clustered_data['cluster']], axis=1))


# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 (Inactifs)')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 (Retraites)')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 (Ma Banque)')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 (CAEL)')
ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n CAEL -- MA Banque")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n Agés -- Jeunes")
#ax.w_zaxis.set_ticklabels([])
plt.legend()
plt.savefig('Plot_3_comp2.png', dpi=600)






# Essai plot 3D interactif
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from plotly.graph_objs import *
plotly.tools.set_credentials_file(username='luciemallet', api_key='EchUlInyh3yStYnkmbhD')

x1, y1, z1 =C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], C[C[:, 10] == 1, 2]
Classe1 = go.Scatter3d(
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
    ), name='Classe 1 (Inactifs)'
)

x2, y2, z2 =C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], C[C[:, 10] == 2, 2]
Classe2 = go.Scatter3d(
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
    ), name='Classe 2 (Retraités)'
)
        
        
x3, y3, z3 = C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], C[C[:, 10] == 3, 2]
Classe3 = go.Scatter3d(
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
    ), name='Classe 3 (Ma Banque)'
)
        
x4, y4, z4 = C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], C[C[:, 10] == 4, 2]
Classe4 = go.Scatter3d(
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
    ), name='Classe 4 (CAEL)'
)
        
data = [Classe1, Classe2, Classe3, Classe4]
layout = go.Layout(
    margin=dict(
        l=0,
        r=0,
        b=0,
        t=0
    ),
    scene=Scene(
        xaxis=XAxis(title='Comp 1 - Activité'),
        yaxis=YAxis(title='Comp 2 - Age'),
        zaxis=ZAxis(title='Comp 3 - CAEL vs Ma Banque ')
    )
    )
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='simple-3d-scatter')



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

plt.bar(r1, y1, width=barWidth, color=['yellow' for i in y1], label='depose')
plt.bar(r2, y2, width=barWidth, color=['pink' for i in y1], label='en ligne')
plt.xticks([r + barWidth for r in range(len(y1))], ['Cluster 1 (Inactifs)', \
'Cluster 2 (Retraités)', 'Cluster 3 (Ma Banque)', 'Cluster 4 (CAEL)'])
plt.suptitle('Proportion de clients ayant souscrits par depose ou en ligne par classe')
plt.legend(loc=2)
plt.savefig('TOP_par_clusters', dpi=600)




# En résumé :

# Pourcentage de TOP dans toute la base
(sum(clustered_data['top_enligne']) / len(clustered_data.axes[0])) * 100 # 0.4 %
(sum(clustered_data['top_depose']) / len(clustered_data.axes[0])) * 100 # 2.96 %

# Pourcentage dans les clusters
mean_top_enligne * 100 # 0.01% pour cluster 1, 0.13% pour cluster 2, 1,3% pour cluster 3 et 0.4% pour cluster 4
mean_top_depose * 100 # 0.17% pour cluster 1, 0.4% pour cluster 2, 9% pour cluster 3 et 4% pour cluster 4






# Import de la base quali pour mozaic plot
base_quali = pd.read_table('C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/base_variables_quali.txt', delimiter=";", dtype={"IDPART_CALCULE":object})
base_quali2 = pd.concat([base_quali,clustered_data['top_enligne'],clustered_data['top_depose'],clustered_data['cluster']],axis=1)

# Code pour mozaic (ne marche pas!!)
#sub = list(range(1, 51, 1))
#plt.figure(figsize=(10, 10))
#var_names = list(base_quali2.columns.values)[1:51]
#for i in range(0, 50):
#    t=mosaic(base_quali2, ['cluster',var_names[i]])[0]
#    plt.subplot(10, 5, sub[i])
#    t


mosaic(base_quali2, ['cluster','Connexion_CAELq'])
mosaic(base_quali2, ['cluster','Connexion_MaBanqueq'])
mosaic(base_quali2, ['cluster','ageq'])
mosaic(base_quali2, ['cluster','dern_entrq'])

mosaic(base_quali2, ['cluster','paiement_carteq'])
mosaic(base_quali2, ['cluster','Agenceq'])
mosaic(base_quali2, ['cluster','Actionsd_CAELq'])
mosaic(base_quali2, ['cluster','virBAMq'])

mosaic(base_quali2, ['cluster','Consult_Comptesq'])
mosaic(base_quali2, ['cluster','revenuq'])
mosaic(base_quali2, ['cluster','top_enligne'])
mosaic(base_quali2, ['cluster','top_depose'])

#au final je trouve les boxplot plus lisibles que les mosaic plot
#quelques variables quali interessantes :
mosaic(base_quali2, ['cluster','libpcs2'])
mosaic(base_quali2, ['cluster','lncsg2'])
mosaic(base_quali2, ['cluster','type_famille'])

# csp par cluster
ctab = pd.crosstab(base_quali2['cluster'],base_quali2['libpcs2']).apply(lambda x: x/x.sum(), axis=1)

ct=ctab.plot( kind='bar', stacked=True, title='Categories socio-professionnelles en proportion par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('lipcs par cluster.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

# segmentation distri par cluster
ctab = pd.crosstab(base_quali2['cluster'],base_quali2['lncsg2']).apply(lambda x: x/x.sum(), axis=1)

ct=ctab.plot( kind='bar', stacked=True, title='Segmentation en proportion par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('lncsg par cluster.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)


# type famille par cluster 
ctab = pd.crosstab(base_quali2['cluster'],base_quali2['type_famille']).apply(lambda x: x/x.sum(), axis=1)

ct=ctab.plot( kind='bar', stacked=True, title='Type de famille en proportion par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('type famille par cluster.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)



# optin sms par cluster 
ctab = pd.crosstab(base_quali2['cluster'],base_quali2['optin_sms']).apply(lambda x: x/x.sum(), axis=1)

ct=ctab.plot( kind='bar', stacked=True, title='Optin sms par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('optin_sms.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)


# optin mail par cluster 
ctab = pd.crosstab(base_quali2['cluster'],base_quali2['optin_mail']).apply(lambda x: x/x.sum(), axis=1)

ct=ctab.plot( kind='bar', stacked=True, title='Optin email par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('optin_mail.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

# Verification des optin sms + mail : les mêmes ?
optin2 = pd.crosstab(base_quali2['optin_sms'].where(base_quali2['cluster']==2),base_quali2['optin_mail'].where(base_quali2['cluster']==2))
optin3 = pd.crosstab(base_quali2['optin_sms'].where(base_quali2['cluster']==3),base_quali2['optin_mail'].where(base_quali2['cluster']==3))
optin4 = pd.crosstab(base_quali2['optin_sms'].where(base_quali2['cluster']==4),base_quali2['optin_mail'].where(base_quali2['cluster']==4))
np.sum(optin4)
base_quali2['optin_sms'].where(base_quali2['cluster']==3).value_counts()
base_quali2['cluster'].value_counts()

# Optin mail et sms par clusters
# Cluster 4 : CAEL
ctab = pd.concat([base_quali2['optin_mail'].where(base_quali2['cluster']==4).value_counts(),base_quali2['optin_sms'].where(base_quali2['cluster']==4).value_counts()],axis=1).apply(lambda x: x/x.sum(), axis=0)
ctab = ctab.transpose()
ctab = pd.concat([ctab['NC'],ctab['OUI'],ctab['NON']],axis=1)
ct=ctab.plot( kind='bar', stacked=True, title="Proportion d'Optin/Optout pour le cluster 4 (CAEL)")
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
#ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('optin_cluster4.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)
# Pie chart
optoutmail = optin4.iloc[:,1]
optoutmail = pd.DataFrame(optoutmail).transpose()
optoutmail = pd.concat([optoutmail['NC'],optoutmail['OUI'],optoutmail['NON']],axis=1)
optoutmail = pd.Series(optoutmail.transpose().iloc[:,0])
ct=optoutmail.plot( kind='pie', autopct='%.0f',title="Proportion d'Optin/Optout sms parmi les Optout mail \n pour le cluster 4 (CAEL)")
ct.set_ylabel('')
plt.savefig('optin_cluster4pie.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

# Cluster 3 : Ma banque
ctab = pd.concat([base_quali2['optin_sms'].where(base_quali2['cluster']==3).value_counts(),base_quali2['optin_mail'].where(base_quali2['cluster']==3).value_counts()],axis=1).apply(lambda x: x/x.sum(), axis=0)
ctab = ctab.transpose()
ctab = pd.concat([ctab['NC'],ctab['OUI'],ctab['NON']],axis=1)
ct=ctab.plot( kind='bar', stacked=True, title="Proportion d'Optin/Optout pour le cluster 3 (Ma Banque)")
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
#ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('optin_cluster3.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)
# Pie chart
optoutsms = optin3.iloc[1,:]
optoutsms = pd.DataFrame(optoutsms).transpose()
optoutsms = pd.concat([optoutsms['NC'],optoutsms['OUI'],optoutsms['NON']],axis=1)
optoutsms = pd.Series(optoutsms.transpose().iloc[:,0])
ct=optoutsms.plot( kind='pie',autopct='%.0f', title="Proportion d'Optin/Optout mail parmi les Optout sms \n pour le cluster 3 (Ma Banque)")
ct.set_ylabel('')
plt.savefig('optin_cluster3pie.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

# Cluster 2 : Retraités
ctab = pd.concat([base_quali2['optin_sms'].where(base_quali2['cluster']==2).value_counts(),base_quali2['optin_mail'].where(base_quali2['cluster']==2).value_counts()],axis=1).apply(lambda x: x/x.sum(), axis=0)
ctab = ctab.transpose()
ctab = pd.concat([ctab['NC'],ctab['OUI'],ctab['NON']],axis=1)
ct=ctab.plot( kind='bar', stacked=True, title="Proportion d'Optin/Optout pour le cluster 2 (Retraités)")
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
#ct.set_xlabel('Classe')
ct.set_xticklabels(ctab.index, rotation=0) 
plt.savefig('optin_cluster2.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)
# Pie chart
optoutsms = optin3.iloc[1,:]
optoutsms = pd.DataFrame(optoutsms).transpose()
optoutsms = pd.concat([optoutsms['NC'],optoutsms['OUI'],optoutsms['NON']],axis=1)
optoutsms = pd.Series(optoutsms.transpose().iloc[:,0])
ct=optoutsms.plot( kind='pie', title="Proportion d'Optin/Optout mail parmi les Optout sms \n pour le cluster 3 (Ma Banque)")
ct.set_ylabel('')
plt.savefig('optin_cluster3pie.png',bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)


# Ajout de test du CHI2
#H0  : independance entre les 2 variables considerees 
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(clustered_data['top_enligne'],clustered_data['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(clustered_data['top_depose'],clustered_data['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['Connexion_CAELq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['Connexion_MaBanqueq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['ageq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['dern_entrq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['paiement_carteq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['Agenceq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['Actionsd_CAELq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['virBAMq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['Consult_Comptesq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['revenuq'],base_quali2['cluster']))
p # Reject H0 : La distribution est différente en fonction des clusters

g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['libsexe'],base_quali2['cluster']))
p # p < 0.05, on rejette H0 
expctd
g, p, dof, expctd = spstat.chi2_contingency(pd.crosstab(base_quali2['libcaju2'],base_quali2['cluster']))
p # p < 0.05, on rejette H0 

