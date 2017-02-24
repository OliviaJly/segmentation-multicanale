# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 16:57:45 2017

@author: Richard
"""


# Librairies utilisées
#from scipy.spatial import distance
from scipy.cluster.vq import vq
import pandas as pd
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import seaborn as sn

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/Historique 3 mois'

## Import des données
# clustering obtenu sur SAS avec var quanti initiales
clusters = pd.read_csv(path + '/CLUST12B_NONT.csv', delimiter=";") 

# clustering obtenu sur SAS avec var quanti tranformées
clusterst = pd.read_csv(path + '/CLUST12B.csv', delimiter=";") 

# Code pour boxplots
sub = list(range(1, 43, 1))
plt.figure(figsize=(43, 43))
var_names = list(clusterst.columns.values)[1:43]
for i in range(0, 42):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=clusterst)
plt.savefig('var x cluster.png', dpi=600)

#calcul des moyennes des indicateurs par cluser
var_means = clusters.groupby('cluster').mean()

# frequence des clusters
freq=pd.DataFrame(clusters)['cluster'].value_counts(sort=False)
freq=pd.DataFrame(freq)
freq['percentage']=freq.apply(lambda r: r/r.sum(), axis=0)
plt.hist(clusterst['cluster'])

# coordonnées des individus dans les 10 premières composantes de l'ACP
coord_ACP_10 = pd.read_csv(path + '/coord_ACP_histo.csv', delimiter=",") 
#fusion avec les clusters
clusters_ACP = pd.concat([coord_ACP_10, clusters['cluster']], axis=1)

#base quanti var transformées au 30/09/2016
base_quantit_0916 = pd.read_csv(path + '/OLIVIA_BASE_QUANTIT_0916.csv', delimiter=";") 
base_quantit_09162 = base_quantit_0916.drop(['IDPART_CALCULE2'], axis=1)

#calcul des coordonnées des ind de 09/16 dans les comp de l'ACP
#nécessite l'exécution de  l'ACP en amont
base_quantit_0916s = pd.DataFrame(scale(base_quantit_09162)) #normalisation des var
coord_ACP_0916 = pca.transform(base_quantit_0916s)
coord_ACP_0916 = pd.DataFrame(coord_ACP_0916)
coord_ACP_0916.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes
coord_ACP_0916_10=coord_ACP_0916.iloc[:, :10] #10 premières composantes
test = -np.array(coord_ACP_0916_10['Comp_3'])
coord_ACP_0916_10 = pd.concat([coord_ACP_0916_10.iloc[:,0:2],pd.DataFrame(test),coord_ACP_0916_10.iloc[:,3:10]],axis=1)
coord_ACP_0916_10 = coord_ACP_0916_10.rename(columns={0: 'Comp_3'})


#calcul des barycentres des clusters dans les comp de l'ACP
clusters_means = clusters_ACP.groupby('cluster').mean()

#calcul des distances entre les barycentres et les individus au 30/09/2016
distance_matrix = vq(coord_ACP_0916_10, clusters_means )
clusters_0916 = pd.DataFrame(distance_matrix[0]) # donne le cluster le + proche pour chaque individu 
clusters_0916 = clusters_0916.rename(columns={0: 'cluster'})
#fusion avec les coord de l'ACP
clusters_0916_ACP = pd.concat([coord_ACP_0916_10, clusters_0916['cluster']], axis=1)

# frequence des clusters
freq_0916=pd.DataFrame(clusters_0916+1)['cluster'].value_counts(sort=False)
freq_0916=pd.DataFrame(freq_0916)
freq_0916['percentage']=freq_0916.apply(lambda r: r/r.sum(), axis=0)
plt.hist(clusters_0916_ACP['cluster'])

# Identification des clusters : 
test = pd.concat([base_quantit_09162,clusters_0916_ACP['cluster']],axis=1)
sub = list(range(1, 43, 1))
plt.figure(figsize=(40, 40))
var_names = list(test.columns.values)[0:42]
for i in range(0, 41):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='cluster', y=var_names[i], data=test)
# OK


#comparaison entre ce clustering et celui calculé uniquement au 30/09/2016
prev_path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'
prev_clusters_0916 = pd.read_csv(prev_path + '/clusters.csv', delimiter=",") 
prev_clusters_0916 = prev_clusters_0916.rename(columns={'0': 'cluster'})

# frequence des clusters
prev_freq_0916=pd.DataFrame(prev_clusters_0916+1)['cluster'].value_counts(sort=False)
prev_freq_0916=pd.DataFrame(prev_freq_0916)
prev_freq_0916['percentage']=prev_freq_0916.apply(lambda r: r/r.sum(), axis=0)
plt.hist(prev_clusters_0916['cluster'])
#comparaison de l'ancien clustering et le nouveau 
prev_clusters_0916['cluster_coded']=prev_clusters_0916['cluster']

#recodage de la variable cluster
#prev_clusters_0916.loc[prev_clusters_0916['cluster']==0,'cluster_coded']=3
#prev_clusters_0916.loc[prev_clusters_0916['cluster']==1,'cluster_coded']=2
#prev_clusters_0916.loc[prev_clusters_0916['cluster']==2,'cluster_coded']=1
#prev_clusters_0916.loc[prev_clusters_0916['cluster']==3,'cluster_coded']=0
crosstab_clus = pd.concat([pd.DataFrame(prev_clusters_0916['cluster_coded'].values+1), 
                                        pd.DataFrame(clusters_0916['cluster'].values+1)],axis=1)
crosstab_clus.columns = ['cluster_sept','cluster_tout_histo'] # Renomer les colonnes
pd.crosstab(crosstab_clus['cluster_tout_histo'],crosstab_clus['cluster_sept'])
pd.crosstab(crosstab_clus['cluster_tout_histo'],crosstab_clus['cluster_sept']).apply(lambda r: r/r.sum(), axis=1)
pd.crosstab(crosstab_clus['cluster_tout_histo'],crosstab_clus['cluster_sept']).apply(lambda r: r/r.sum(), axis=0)
       
## Représentation dans le plan de l'ACP des clusters
clusters_0916_ACP['cluster']=clusters_0916_ACP['cluster']+1
C=np.array(clusters_0916_ACP)
# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 Inactifs')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 Retraités')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 Ma banque')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 CAEL')
ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n  Ma Banque -- CAEL")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n Agés -- Jeunes")
#ax.w_zaxis.set_ticklabels([])
plt.legend()
plt.savefig('Plot_3D.png', dpi=600)

#on regarde plus en détail les Ma Banque devenus retraités 
def codage(x):
    if x['prev_cluster']==2 and x['cluster']==1 :
       return 4
    else :
        return x['cluster']

compar_clusters=pd.concat([prev_clusters_0916['cluster_coded'],clusters_0916],axis=1)
compar_clusters.columns=['prev_cluster','cluster'] 
compar_clusters['cluster2']=compar_clusters['cluster']
compar_clusters['cluster2']=compar_clusters.apply(codage,axis=1)

# Recuperation des variables initiales pour voir les comportements de la classe MB-Retraités
base_clus = pd.concat([base_quantit_09162,compar_clusters['cluster2']],axis=1)

# Boxplot nouvelle pred
sub = list(range(1, 43, 1))
plt.figure(figsize=(40, 40))
var_names = list(base_clus.columns.values)[0:42]
for i in range(0, 41):
    plt.subplot(9, 5, sub[i])
    sn.boxplot(x='cluster2', y=var_names[i], data=base_clus)




## Représentation dans le plan de l'ACP des clusters
clusters_0916_ACP2=clusters_0916_ACP.copy()
clusters_0916_ACP2['cluster']=compar_clusters['cluster2']+1
C=np.array(clusters_0916_ACP2)
# PLOT 3D
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(C[C[:, 10] == 1, 2], C[C[:, 10] == 1, 0], C[C[:, 10] == 1, 1], \
           c='royalblue', cmap=plt.cm.Paired, label='Classe 1 Inactifs')
ax.scatter(C[C[:, 10] == 2, 2], C[C[:, 10] == 2, 0], C[C[:, 10] == 2, 1], \
           c='forestgreen', cmap=plt.cm.Paired, label='Classe 2 Retraités')
ax.scatter(C[C[:, 10] == 3, 2], C[C[:, 10] == 3, 0], C[C[:, 10] == 3, 1], \
           c='firebrick', cmap=plt.cm.Paired, label='Classe 3 MB')
ax.scatter(C[C[:, 10] == 4, 2], C[C[:, 10] == 4, 0], C[C[:, 10] == 4, 1], \
           c='slateblue', cmap=plt.cm.Paired, label='Classe 4 CAEL')
ax.scatter(C[C[:, 10] == 5, 2], C[C[:, 10] == 5, 0], C[C[:, 10] == 5, 1], \
           c='blue', cmap=plt.cm.Paired, label='Classe 5 (Ma Banque devenus Retraités)')


ax.set_title("Représentation des classes d'invididus")
ax.set_xlabel("\n Composante 3 \n  Ma Banque -- CAEL")
#ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("\n Composante 1 \n Activité -- Inactivité")
#ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("\n Composante 2 \n Agés -- Jeunes")
#ax.w_zaxis.set_ticklabels([])
plt.legend()
plt.savefig('Plot_3D.png', dpi=600)


# Essai plot 3D interactif
import plotly
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
    ), name='Classe 1 Inactifs'
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
    ), name='Classe 2 Retraités'
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
    ), name='Classe 3 MB'
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
    ), name='Classe 4 CAEL'
)
     
x5, y5, z5 = C[C[:, 10] == 5, 0], C[C[:, 10] == 5, 1], C[C[:, 10] == 5, 2]
Classe5 = go.Scatter3d(
    x=x5,
    y=y5,
    z=y5,
    mode='markers',
    marker=dict(
        color='orange',
        size=5,
        symbol='triangle-up',
        line=dict(
            color='orange',
            width=1
        ),
        opacity=0.8
    ), name='Classe 5 (Ma Banque devenus retraités)'
)
     
data = [Classe1, Classe2, Classe3, Classe4, Classe5]
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



#### 06/16

#base quanti var transformées 
base_quantit_0616 = pd.read_csv(path + '/OLIVIA_BASE_QUANTIT_0616.csv', delimiter=";") 
base_quantit_06162 = base_quantit_0616.drop(['IDPART_CALCULE2'], axis=1)

#calcul des coordonnées des ind dans les comp de l'ACP
#nécessite l'exécution de  l'ACP en amont
base_quantit_0616s = pd.DataFrame(scale(base_quantit_06162)) #normalisation des var
coord_ACP_0616 = pca.transform(base_quantit_0616s)
coord_ACP_0616 = pd.DataFrame(coord_ACP_0616)
coord_ACP_0616.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes
coord_ACP_0616_10=coord_ACP_0616.iloc[:, :10] #10 premières composantes

#calcul des distances entre les barycentres et les individus 
distance_matrix_0616 = vq(coord_ACP_0616_10, clusters_means )
clusters_0616 = pd.DataFrame(distance_matrix_0616[0]) # donne le cluster le + proche pour chaque individu 
clusters_0616 = clusters_0616.rename(columns={0: 'cluster'})
#fusion avec les coord de l'ACP
clusters_0616_ACP = pd.concat([coord_ACP_0616_10, clusters_0616['cluster']], axis=1)

# frequence des clusters
freq_0616=pd.DataFrame(clusters_0616+1)['cluster'].value_counts(sort=False)
freq_0616 = pd.DataFrame(freq_0616)
freq_0616['percentage']=freq_0616.apply(lambda r: r/r.sum(), axis=0)



#### 03/16

#base quanti var transformées 
base_quantit_0316 = pd.read_csv(path + '/OLIVIA_BASE_QUANTIT_0316.csv', delimiter=";") 
base_quantit_03162 = base_quantit_0316.drop(['IDPART_CALCULE2'], axis=1)

#calcul des coordonnées des ind dans les comp de l'ACP
#nécessite l'exécution de  l'ACP en amont
base_quantit_0316s = pd.DataFrame(scale(base_quantit_03162)) #normalisation des var
coord_ACP_0316 = pca.transform(base_quantit_0316s)
coord_ACP_0316 = pd.DataFrame(coord_ACP_0316)
coord_ACP_0316.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes
coord_ACP_0316_10=coord_ACP_0316.iloc[:, :10] #10 premières composantes

#calcul des distances entre les barycentres et les individus 
distance_matrix_0316 = vq(coord_ACP_0316_10, clusters_means )
clusters_0316 = pd.DataFrame(distance_matrix_0316[0]) # donne le cluster le + proche pour chaque individu 
clusters_0316 = clusters_0316.rename(columns={0: 'cluster'})
#fusion avec les coord de l'ACP
clusters_0316_ACP = pd.concat([coord_ACP_0316_10, clusters_0316['cluster']], axis=1)

# frequence des clusters
freq_0316=pd.DataFrame(clusters_0316+1)['cluster'].value_counts(sort=False)
freq_0316 = pd.DataFrame(freq_0316)
freq_0316['percentage']=freq_0316.apply(lambda r: r/r.sum(), axis=0)

#### 12/15

#base quanti var transformées 
base_quantit_1215 = pd.read_csv(path + '/OLIVIA_BASE_QUANTIT_1215.csv', delimiter=";") 
base_quantit_12152 = base_quantit_1215.drop(['IDPART_CALCULE2'], axis=1)

#calcul des coordonnées des ind dans les comp de l'ACP
#nécessite l'exécution de  l'ACP en amont
base_quantit_1215s = pd.DataFrame(scale(base_quantit_12152)) #normalisation des var
coord_ACP_1215 = pca.transform(base_quantit_1215s)
coord_ACP_1215 = pd.DataFrame(coord_ACP_1215)
coord_ACP_1215.columns = ["Comp_" + str(l) for l in list(range(1, 43, 1))] # Renomer les colonnes
coord_ACP_1215_10=coord_ACP_1215.iloc[:, :10] #10 premières composantes

#calcul des distances entre les barycentres et les individus 
distance_matrix_1215 = vq(coord_ACP_1215_10, clusters_means )
clusters_1215 = pd.DataFrame(distance_matrix_1215[0]) # donne le cluster le + proche pour chaque individu 
clusters_1215 = clusters_1215.rename(columns={0: 'cluster'})
#fusion avec les coord de l'ACP
clusters_1215_ACP = pd.concat([coord_ACP_1215_10, clusters_1215['cluster']], axis=1)

# frequence des clusters
freq_1215=pd.DataFrame(clusters_1215+1)['cluster'].value_counts(sort=False)
freq_1215 = pd.DataFrame(freq_1215)
freq_1215['percentage']=freq_1215.apply(lambda r: r/r.sum(), axis=0)

#fusion des 4 freq 
evol_freq = pd.concat([freq_1215['cluster'], freq_0316['cluster'], freq_0616['cluster'], freq_0916['cluster']], axis=1)
evol_freq.columns = ["freq_1215", "freq_0316", "freq_0616", "freq_0916"] # Renomer les colonnes

#plot
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.dates import MonthLocator, DateFormatter

dates=["30/12/2015","30/03/2016","30/06/2016", "30/09/2016"]
x = [dt.datetime.strptime(d,'%d/%m/%Y').date() for d in dates]

# every 3rd month
months = MonthLocator(range(1,13), bymonthday=30,interval=3)
monthsFmt = DateFormatter("%b '%y")

fig, ax = plt.subplots()
ax.plot(x, evol_freq.loc[1,:], 'b', x, evol_freq.loc[2,:], 'bs', x, evol_freq.loc[3,:], 'g^',x, evol_freq.loc[4,:], 'r--')
ax.xaxis.set_major_locator(months)
ax.xaxis.set_major_formatter(monthsFmt)
ax.autoscale_view()

ax.grid(True)
fig.autofmt_xdate()
plt.show()

#matrices de transition
pd.crosstab(clusters_1215['cluster'].values+1,clusters_0316['cluster'].values+1)
#pourcentage par ligne 
pd.crosstab(clusters_1215['cluster'].values+1,clusters_0316['cluster'].values+1).apply(lambda r: r/r.sum(), axis=1)

pd.crosstab(clusters_0316['cluster'].values+1,clusters_0616['cluster'].values+1)
#pourcentage par ligne 
pd.crosstab(clusters_0316['cluster'].values+1,clusters_0616['cluster'].values+1).apply(lambda r: r/r.sum(), axis=1)

pd.crosstab(clusters_0616['cluster'].values+1,clusters_0916['cluster'].values+1)
#pourcentage par ligne 
pd.crosstab(clusters_0616['cluster'].values+1,clusters_0916['cluster'].values+1).apply(lambda r: r/r.sum(), axis=1)

