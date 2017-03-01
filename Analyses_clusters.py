# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 17:34:42 2017

@author: Lucie
"""



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
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
from plotly.graph_objs import *

# Definition du chemin où sont situées les données :
PATH = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'


# Import des résultats du clustering
pred = pd.read_csv(PATH + '/clusters.csv', delimiter=",") 
# Import des données initiales
quanti_trans = pd.read_csv(PATH + '/quanti_trans2.csv', delimiter=",", \
                           dtype={"IDPART_CALCULE":object})
base_test2 = quanti_trans.drop(['IDPART_CALCULE', 'Actionsd_MaBanque_3m', \
                                'Lecture_mess_3m', 'Ecriture_mess_3m'], 1) 
# Base quanti avant transformation des variables
base_quanti = pd.read_csv(PATH + '/base_quanti.csv', delimiter=",", dtype={"IDPART_CALCULE":object})
# Base coordonnées ACP
data_coor2 = pd.read_csv(PATH + '/PCA_coor2.csv', delimiter=",") 
data_coor3 = data_coor2.iloc[:, :10]
del data_coor2

# Frequence des clusters
count=pd.DataFrame(pred+1)['0'].value_counts(sort=False)
count=pd.DataFrame(count)

# now to plot the figure...
plt.figure(figsize=(12, 8))
ax = count.plot(kind='bar')
ax.set_title("Nb d'individus par classe")
ax.set_xlabel("Classe")
ax.set_ylabel("Nb d'individus")
rects = ax.patches
# Now make some labels
labels = [count.iat[i,0] for i in range(len(rects))] 
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')
plt.savefig('frequence clusters.png', dpi=600) 
del label, labels, height, count


# Concatenation avec les variables initiales transformées
clustered_data = pd.concat([base_test2, pd.DataFrame(pred)], axis=1)
clustered_data = clustered_data.rename(columns={'0': 'cluster'}) # Attention au type du nom de cluster
clustered_data['cluster']=clustered_data['cluster']+1


# Concatenation avec les variables initiales non transformées
clustered_data2 = pd.concat([base_quanti, pd.DataFrame(pred)], axis=1)
clustered_data2 = clustered_data2.rename(columns={'0': 'cluster'})
clustered_data2['cluster']=clustered_data2['cluster']+1
   

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
del C


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
del r2, y1, y2, barWidth


# Pourcentage de TOP dans toute la base
(sum(clustered_data['top_enligne']) / len(clustered_data.axes[0])) * 100 # 0.4 %
(sum(clustered_data['top_depose']) / len(clustered_data.axes[0])) * 100 # 2.96 %

# Pourcentage dans les clusters
mean_top_enligne * 100 # 0.01% pour cluster 1, 0.13% pour cluster 2, 1,3% pour cluster 3 et 0.4% pour cluster 4
mean_top_depose * 100 # 0.17% pour cluster 1, 0.4% pour cluster 2, 9% pour cluster 3 et 4% pour cluster 4
del mean_top_enligne, mean_top_depose


# Import de la base quali pour mozaic plot
base_quali = pd.read_table('C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2/base_variables_quali.csv', delimiter=";", dtype={"IDPART_CALCULE":object})
base_quali2 = pd.concat([base_quali,clustered_data['top_enligne'],clustered_data['top_depose'],clustered_data['cluster']],axis=1)

# Variables quali interessantes :
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
del optin2, optin3, optin4, optoutmail, optoutsms, g, dof, ctab, p


## ANALYSE DE LA CLASSE DES INACTIFS (cluster 1)
# Croisement clustering et segmentation comportementale (fidelité)
segm_fidelite = pd.read_csv('C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/OLIVIA_FIDELITE_ECHANTILLON.csv', delimiter=";") 

base_quali2 = pd.concat([base_quali2,segm_fidelite['SGMT_FIDELITE']],axis=1)

cros = pd.crosstab(base_quali2['cluster'],base_quali2['SGMT_FIDELITE'])
cros_pourcent = pd.crosstab(base_quali2['cluster'],base_quali2['SGMT_FIDELITE']).apply(lambda r: r/r.sum(), axis=1)

ct=cros_pourcent.plot( kind='bar', stacked=True, title='Segment fidelité par classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Proportion')
ct.set_xlabel('Classe')
ct.set_xticklabels(cros_pourcent.index, rotation=0) 
plt.savefig('Segment fidelité par cluster.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

# Croisement clustering et cibles attrition 
mdc_attri = pd.read_csv(PATH+'/MDC_ATTRITION_09163.csv', delimiter=";", encoding="ISO-8859-1", \
                        dtype={"ID_PART":object})

#creation des var quali Multibancarise, Clientnonvu, Baissefluxope
#Si idpart ciblé par le mdc alors = 1, sinon = nan
mdc_attri.loc[(mdc_attri['COL1']=='CLIENT NON VU')|(mdc_attri['COL2']=='CLIENT NON VU'),"Clientnonvu"]=1
mdc_attri.loc[(mdc_attri['COL1']=='BAISSE FLUX OPE')|(mdc_attri['COL2']=='BAISSE FLUX OPE'),"Baissefluxope"]=1
mdc_attri.loc[(mdc_attri['COL1']=='MULTIBANCARISE')|(mdc_attri['COL2']=='MULTIBANCARISE'),"Multibancarise"]=1

mdc_attri2 = mdc_attri[['ID_PART','Multibancarise','Baissefluxope','Clientnonvu']]
mdc_attri2=mdc_attri2.rename(columns={'ID_PART':'IDPART_CALCULE'})
mdc_attri3 = mdc_attri2.drop('IDPART_CALCULE', axis=1)  


#compte d'individus ciblés sur la base totale de clients 
count_tot = mdc_attri3.apply(pd.value_counts)
count_tot = count_tot.transpose()
count_tot = pd.DataFrame(count_tot).rename(columns={1.0: 'total'})

#jointure sur base quali 
base_quali3=base_quali2.join(mdc_attri2.set_index('IDPART_CALCULE'), how='left', on='IDPART_CALCULE')

#remplace les nan par 0
base_quali3.loc[pd.isnull(base_quali3['Multibancarise']),'Multibancarise']=0
base_quali3.loc[pd.isnull(base_quali3['Clientnonvu']),'Clientnonvu']=0
base_quali3.loc[pd.isnull(base_quali3['Baissefluxope']),'Baissefluxope']=0

mdc_ech_idpart=base_quali3[['IDPART_CALCULE','Multibancarise','Baissefluxope','Clientnonvu','cluster']]

#compte d'individus ciblés sur l'echantillon en 09/16
mdc_ech=base_quali3[['Multibancarise','Baissefluxope','Clientnonvu']]
count_ech=mdc_ech.apply(pd.value_counts)
count_ech=count_ech.iloc[1]
count_ech = pd.DataFrame(count_ech).rename(columns={1.0: 'ech'})

#Chaque MDC est représenté avec env la meme proportion dans l'echantillon
mdc_compar=pd.concat([count_tot,count_ech], axis=1)
mdc_compar['percent ech/total']=mdc_compar['ech']/mdc_compar['total']

# MDC attrition par cluster 
cross_tab=mdc_ech_idpart.groupby(['cluster']).sum()
cross_tab_percent=cross_tab.apply(lambda r: r/r.sum(),axis=0)    
            
cross_tab['tot col']=cross_tab.apply(sum,axis=1)
cross_tab['cluster size']= mdc_ech_idpart['cluster'].value_counts()
cross_tab.loc['tot row']=cross_tab.apply(sum,axis=0)
cross_tab_percent['cibles / total']=cross_tab['tot col']/cross_tab['cluster size']*100
cross_tab_test=cross_tab.iloc[:4,:3]
cross_tab_test.index=(['Inactifs','Agence','Ma Banque', 'CAEL'])

#graph 
ct=cross_tab_test.plot( kind='bar', stacked=True, title='Nb de clients ciblés par les MDC attrition dans chaque classe')
lgd=ct.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
ct.set_ylabel('Nb de clients')
ct.set_xlabel('Classe')
ct.set_xticklabels(cross_tab_test.index, rotation=0) 

rects=ct.patches
rects2=cross_tab['tot col']
rects2.index=[1,2,3,4,5]
rects2=rects2.loc[:4]

labels=cross_tab_percent['cibles / total']
labels2 = [ str(i)[:3]+"%" for i in labels]

for rect, label, rect2 in zip(rects, labels2, rects2):
    height = rect2
    ct.text(rect.get_x() + rect.get_width()/2, height+2 , label, ha='center', va='bottom')

plt.savefig('Clusters x MDC attrition.png',  bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)
