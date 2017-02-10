# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:08:12 2017

@author: Lucie
"""

##  Copier le code dans la console python

########### IMPORT DES DONNEES :


# Librairies utilisées
import pandas as pd
import numpy as np
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
import matplotlib.pyplot as plt
import scipy.stats
import seaborn as sn

# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données/v2'

# Import des données
df = pd.read_csv(path +'/OLIVIA_BASE_PART_092016.csv', nrows=20000, delimiter=";",
                 encoding="ISO-8859-1",
                 dtype={"IDPART_CALCULE":object, "IDCLI_CALCULE":object})
# encoding = "ISO-8859-1" est nécessaire, le défaut ne fonctionne pas (impossible
# de lire les données en utf-8,je ne sais pas trop pourquoi
# dtype change le type de variables, object = charactère


############ METTRE LES DONNEES SOUS LE BON FORMAT :

# Changement du format de date_part en date:
df["date_part"] = pd.to_datetime(df["date_part"])

# Verification du type des données :
types = df.dtypes  # Ok. (Float : int avec décimales)




############ MMISE EN FORME DE LA BASE D'ETUDE (ACP) :

# Selection des variables quanti + id part dans une nouvelle table df_quanti :
df_quanti = pd.concat([df['IDPART_CALCULE'],
                       df.select_dtypes(include=['float64', 'int64'])], axis=1)
# On garde seulement les variables quanti (float et int), et on fait un cbind
# (axis=1 veut dire qu'on concatene sur les colonnes


#distribution des var quali top_depose et top_enligne
count_depose = df_quanti['top_depose'].value_counts()
plt.pie(count_depose, colors=['lightskyblue', 'gold'], labels=['0', '1'], \
        autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.suptitle('Distribution de la variable top depose')
plt.savefig('distrib_topdepose.png', dpi=600)
plt.show()

count_depose = df_quanti['top_enligne'].value_counts()
plt.pie(count_depose, colors=['lightskyblue', 'gold'], labels=['0', '1'], \
        autopct='%1.1f%%', startangle=90)
plt.axis('equal')
plt.suptitle('Distribution de la variable top en ligne')
plt.savefig('distrib_topenligne.png', dpi=600)
plt.show()

#on croise top depose et top en ligne avec les connexions CAEL et Ma Banque
sn.boxplot(x='top_depose', y='Connexion_MaBanque_3m', data=df_quanti)
sn.boxplot(x='top_depose', y='Connexion_CAEL_3m', data=df_quanti)
sn.boxplot(x='top_enligne', y='Connexion_MaBanque_3m', data=df_quanti)
sn.boxplot(x='top_enligne', y='Connexion_CAEL_3m', data=df_quanti)


moy = df_quanti.groupby('top_depose')['Connexion_MaBanque_3m', 'Connexion_CAEL_3m'].mean()

#barplot
barWidth = 0.4
y1 = moy.ix[0]
y2 = moy.ix[1]
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, y1, width=barWidth, color=['yellow' for i in y1], label='top depose 0')
plt.bar(r2, y2, width=barWidth, color=['pink' for i in y1], label='top depose 1')
plt.xticks([r + barWidth for r in range(len(y1))], ['Connex MaBanque', 'Connex CAEL'])

plt.suptitle('Nb de connexions moyen Ma Banque et CAEL par catégorie de top depose')
plt.legend()
plt.savefig('connexions_vs_topdepose.png', dpi=600)


medi = df_quanti.groupby('top_depose')['Connexion_MaBanque_3m', 'Connexion_CAEL_3m'].median()
# + de connexions MB et CAEL pour les deposes et enligne

# Check distribution des nb depose et enligne
layout = dict(autosize=True)
plt.hist(np.array(df_quanti['nb_contrats_depose']))
plt.hist(np.array(df_quanti['nb_contrats_enligne']))



###### Selection des variables interessantes :

### Code suppression des variables sur 1 et 2 mois :

var_names = list(df_quanti.columns.values)   # Liste contenant les noms des variables
sub_1m = '_1m' # Pattern à supprimer
sub_2m = '_2m' # Pattern à supprimer

x = list(range(0, len(var_names)))        # Liste qui contiendra des booleens
for i in range(0, len(var_names)):
    if sub_1m in var_names[i] or sub_2m in var_names[i]:
        x[i] = False
    else:
        x[i] = True
# x = True quand le nom de la variable ne contient aucun des pattern.
# On gardera donc les variables quand x = True

var = pd.DataFrame({'vf': x, 'noms': var_names}) # Creation d'un data frame intermédiaire
var = var[var.vf == False] # On garde seulement les variables quand c'est False
# parce que panda n'a pas de fonction keep, mais uniquement de fonction drop

var_drop = list(var['noms']) # Extraction de la liste des variables dont on veut se debarasser
var_drop.append('top_depose') #ajout des variables top depose et top en ligne qui sont des var quali
var_drop.append('top_enligne')

base_quanti = df_quanti.drop(var_drop, axis=1)   # Drop les variables qu'on ne veut plus
### Fin code suppression des variables sur 1 et 2 mois

del(df, df_quanti, i, sub_1m, sub_2m, types, var, var_drop, var_names, x)



### Recoder les variables comme dans le code SAS (transformations monotones)
base_quanti2 = base_quanti.copy()

base_quanti2['MT_OPERATION_DEPOT_3m'] = np.log(base_quanti2['MT_OPERATION_DEPOT_3m']+ 1)
base_quanti2['mt_paiement_chq_3m'] = np.log(-base_quanti2['mt_paiement_chq_3m']+ 1)
base_quanti2['mt_paiement_carte_3m'] = np.log(-base_quanti2['mt_paiement_carte_3m']+ 1)
base_quanti2['REVENU_EST_MM'] = np.log(base_quanti2['REVENU_EST_MM']+ 1)
base_quanti2['DEPENSES_RECURRENTES_EST_M'] = np.log(base_quanti2['DEPENSES_RECURRENTES_EST_M']+ 1)
base_quanti2['SURFACE_FINANCIERE'] = np.log(base_quanti2['SURFACE_FINANCIERE'] - min(base_quanti2['SURFACE_FINANCIERE']) + 1)
base_quanti2['ENCOURS_DAV'] = np.log(base_quanti2['ENCOURS_DAV']+ 1)
base_quanti2['Agence_3m'] = np.sqrt(base_quanti2['Agence_3m'])
base_quanti2['Agence_vente_3m'] = np.sqrt(base_quanti2['Agence_vente_3m'])
base_quanti2['Agence_retrait_3m'] = np.sqrt(base_quanti2['Agence_retrait_3m'])
base_quanti2['Agence_depot_3m'] = np.sqrt(base_quanti2['Agence_depot_3m'])
base_quanti2['Agence_rdv_3m'] = np.sqrt(base_quanti2['Agence_rdv_3m'])
base_quanti2['Agence_vir_3m'] = np.sqrt(base_quanti2['Agence_vir_3m'])
base_quanti2['Connexion_CAEL_3m'] = np.log(base_quanti2['Connexion_CAEL_3m']+ 1)
base_quanti2['Connexion_MaBanque_3m'] = np.log(base_quanti2['Connexion_MaBanque_3m']+ 1)
base_quanti2['Actions_CAEL_3m'] = np.log(base_quanti2['Actions_CAEL_3m']+ 1)
base_quanti2['Actions_MaBanque_3m'] = np.log(base_quanti2['Actions_MaBanque_3m']+ 1)
base_quanti2['Lecture_mess_3m'] = np.sqrt(base_quanti2['Lecture_mess_3m'])
base_quanti2['Ecriture_mess_3m'] = np.sqrt(base_quanti2['Ecriture_mess_3m'])
base_quanti2['Ges_Alerte_CAEL_3m'] = np.sqrt(base_quanti2['Ges_Alerte_CAEL_3m'])
base_quanti2['Consult_BGPI_3m'] = np.sqrt(base_quanti2['Ges_Alerte_CAEL_3m'])
base_quanti2['Ges_Alerte_CAEL_3m'] = np.sqrt(base_quanti2['Consult_carte_3m'])
base_quanti2['Consult_CAtitre_3m'] = np.sqrt(base_quanti2['Consult_CAtitre_3m'])
base_quanti2['Consult_credit_3m'] = np.log(base_quanti2['Consult_credit_3m']+1)
base_quanti2['Consult_epargne_3m'] = np.log(base_quanti2['Consult_epargne_3m']+1)
base_quanti2['Dem_contact_3m'] = np.sqrt(base_quanti2['Dem_contact_3m'])
base_quanti2['Consult_PACIFICA_3m'] = np.sqrt(base_quanti2['Consult_PACIFICA_3m'])
base_quanti2['Consult_PREDICA_3m'] = np.sqrt(base_quanti2['Consult_PREDICA_3m'])
base_quanti2['Simul_creditconso_3m'] = np.sqrt(base_quanti2['Simul_creditconso_3m'])
base_quanti2['Simul_credithabitat_3m'] = np.sqrt(base_quanti2['Simul_credithabitat_3m'])
base_quanti2['Consult_Sofinco_3m'] = np.sqrt(base_quanti2['Consult_Sofinco_3m'])
base_quanti2['Vir_BAM_3m'] = np.sqrt(base_quanti2['Vir_BAM_3m'])
base_quanti2['Consult_Comptes_3m'] = np.log(base_quanti2['Consult_Comptes_3m']+1)


# Test transformations pour scatter plots
base_quanti.boxplot('Connexion_CAEL_3m')
base_quanti.boxplot('Connexion_MaBanque_3m')

# Centrer et réduire les variables
base_quanti2['Connexion_CAEL_CR'] = (base_quanti['Connexion_CAEL_3m'] - np.mean(base_quanti['Connexion_CAEL_3m'])) / np.std(base_quanti['Connexion_CAEL_3m'])
base_quanti2['Connexion_MABANQUE_CR'] = (base_quanti['Connexion_MaBanque_3m'] - np.mean(base_quanti['Connexion_MaBanque_3m'])) / np.std(base_quanti['Connexion_MaBanque_3m'])
base_quanti2['Actions_CAEL_CR'] = (base_quanti['Actions_CAEL_3m'] - np.mean(base_quanti['Actions_CAEL_3m'])) / np.std(base_quanti['Actions_CAEL_3m'])
base_quanti2['Actions_MABANQUE_CR'] = (base_quanti['Actions_MaBanque_3m'] - np.mean(base_quanti['Actions_MaBanque_3m'])) / np.std(base_quanti['Actions_MaBanque_3m'])



 # calcul des correlations
base_corr = base_quanti2.drop(['IDPART_CALCULE'], axis=1)
mat_corr = np.corrcoef(base_corr, rowvar=0) #matrice des correlations (pearson)
names = list(base_corr.columns.values)
mat_corr = pd.DataFrame(mat_corr, index=names, columns=names)
#ajout des noms aux lignes et colonnes
# Corr nb_contrats_depose mabanque = 0.14
# Corr nb_contrats_depose cael = 0.13

scipy.stats.spearmanr(base_quanti2['Connexion_MaBanque_3m'], base_quanti2['nb_contrats_depose'])
scipy.stats.spearmanr(base_quanti2['Connexion_CAEL_3m'], base_quanti2['nb_contrats_depose'])

#nuage de points entre var quanti

## 1ere methode : utilise le package plotly, pas vraiment d'intérêt pour les graphiques simples

plotly.tools.set_credentials_file(username='oliviaJLY', api_key='BCwodb0FVCOegRrBDiST')

# Create a trace
trace = go.Scatter(
    x=base_quanti['DEPENSES_RECURRENTES_EST_M'],
    y=base_quanti['REVENU_EST_MM'],
    mode='markers',
    marker=dict(
        color='FFBAD2',
        line=dict(width=1)
    )
)

layout = go.Layout(
    xaxis=dict(title='Dépenses', range=[0, 30000]),
    yaxis=dict(title='Revenus', range=[0, 30000]),
    title='Revenus et dépenses mensuels des clients')

data = [trace]
fig = go.Figure(data=data, layout=layout)
# Plot and embed in ipython notebook!
py.iplot(fig, filename='basic-scatter')
del(trace, names, layout, data, fig)


## 2eme methode : avec le package matplotlib

#Scatter plot depenses et revenus
x = base_quanti['DEPENSES_RECURRENTES_EST_M']
y = base_quanti['REVENU_EST_MM']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
# Chart title
plt.title('Revenus et dépenses mensuels des clients')
# y label
plt.ylabel('Revenus')
# x label
plt.xlabel('Dépenses')
# set the figure boundaries
plt.xlim(0, 20000)
plt.ylim(0, 20000)
plt.savefig('Dep_revenus.png', dpi=600)
# OK


#Actions CAEL et Actions Ma Banque sur var transformées
x = base_corr['Actions_CAEL_3m']
y = base_corr['Actions_MaBanque_3m']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Nb d'actions CAEL en fct du nb d'actions Ma Banque")
# y label
plt.ylabel('Actions Ma Banque')
# x label
plt.xlabel('Actions CAEL')
# set the figure boundaries
plt.savefig('Actions_CAEL_MABANQUE.png', dpi=600)
# Mieux en transformées


#Actions CAEL et Actions Ma Banque sur var non transformées
x = base_quanti['Actions_CAEL_3m']
y = base_quanti['Actions_MaBanque_3m']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Nb d'actions CAEL en fct du nb d'actions Ma Banque")
# y label
plt.ylabel('Actions Ma Banque')
# x label
plt.xlabel('Actions CAEL')
# set the figure boundaries
plt.show()


#Connexions CAEL et connexions Ma Banque sur var transformées CR
x = base_quanti2['Connexion_CAEL_CR']
y = base_quanti2['Connexion_MABANQUE_CR']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Nb de connexions CAEL en fct du nb de connexions Ma Banque")
# y label
plt.ylabel('Connexions Ma Banque')
# x label
plt.xlabel('Connexions CAEL')
# set the figure boundaries
plt.show()

#Connexions CAEL et connexions Ma Banque sur var transformées log
x = base_quanti2['Connexion_CAEL_3m']
y = base_quanti2['Connexion_MaBanque_3m']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Nb de connexions CAEL en fct du nb de connexions Ma Banque")
# y label
plt.ylabel('Connexions Ma Banque')
# x label
plt.xlabel('Connexions CAEL')
# set the figure boundaries
plt.savefig('Connexions_CAEL_MABANQUE.png', dpi=600)
# Mieux en log


#Connexion Ma Banque et age sur base non transformée
x = base_quanti['age']
y = base_quanti['Connexion_MaBanque_3m']


scipy.stats.spearmanr(x, y) #correlation spearman (non parametrique), moins sensible aux outliers

# x = base_corr['age']
# y = base_corr['Connexion_MaBanque_3m']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Connexions à Ma Banque en fct de l'âge")
# y label
plt.ylabel('Connexions Ma Banque')
# x label
plt.xlabel('Age')
# set the figure boundaries
plt.savefig('figure_1.png', dpi=600)

#Connexion Ma Banque et age sur base transformée
x = base_quanti2['age']
y = base_quanti2['Connexion_MaBanque_3m']


scipy.stats.spearmanr(x, y) #correlation spearman (non parametrique), moins sensible aux outliers

# x = base_corr['age']
# y = base_corr['Connexion_MaBanque_3m']

plt.style.use('ggplot')
plt.plot(x, y, 'o')
plt.xticks([], []) #enleve l'échelle des axes
plt.yticks([], [])
# Chart title
plt.title("Connexions à Ma Banque en fct de l'âge")
# y label
plt.ylabel('Connexions Ma Banque')
# x label
plt.xlabel('Age')
# set the figure boundaries
plt.savefig('Connexions_MABANQUE_AGE.png', dpi=600)
# Mieux en transformée

# Pas intéressant de regarder les Scatters plots entre les dépose/ en ligne avec CAEL et ma banque :
# trop peu d'observations sur les personnes qui ont fait plusieurs déposes


# Retrait des variables correlees et non discriminantes
base_quanti2 = base_quanti2.drop(['DEPENSES_RECURRENTES_EST_M', 'SMS_recus_3m',
                                  'SMS_authentif_3m', 'Automate_depot_3m', 'Automate_retrait_3m',
                                  'Duree_CAEL_3m', 'Duree_MaBanque_3m', 'Actions_CAEL_3m',
                                  'Actions_MaBanque_3m', 'Ges_Alerte_CAEL_3m', 'Consult_BGPI_3m',
                                  'Consult_carte_3m', 'Dem_contact_3m', 'Simul_creditconso_3m',
                                  'Simul_credithabitat_3m', 'Consult_Sofinco_3m', 'Email_3m',
                                  'Appels_entrants_3m', 'Connexion_CAEL_CR', 'Connexion_MABANQUE_CR',
                                  'Actions_CAEL_CR', 'Actions_MABANQUE_CR'], axis=1)


# base_quanti2 est la même base d'étude que pour l'ACP sous SAS


# Enregistre la base en csv :
base_quanti2.to_csv(path + '/quanti_trans2.csv', index=False)

