# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:04:25 2017

@author: Lucie
"""

########## Identification des outliers



## Librairies utilisées
import pandas as pd
import numpy as np
import pylab
import matplotlib.pyplot as plt







# Import de la base des données quanti avant transformation des variables


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale2/Données'  

# Import des données
quanti = pd.read_csv(path + '/quanti.csv',delimiter=",",dtype={"IDPART_CALCULE":object})


quanti = quanti.drop(['IDPART_CALCULE'],axis=1) # On enleve ID_part pour avoir seulement

quanti['mt_paiement_chq_3m'] = -quanti['mt_paiement_chq_3m']
quanti['mt_paiement_carte_3m'] = -quanti['mt_paiement_carte_3m']
# les variables quanti


_, bp = pd.DataFrame.boxplot(quanti, return_type='both') # Creation d'un objet boxplot qui
# identifie les outliers
outliers = [flier.get_ydata() for flier in bp["fliers"]] # Recuperation des outliers
            
# boucle pour creer une nouvelle base sans outliers base_quanti_out
var_names = list(quanti.columns.values)
quanti_out = quanti
for column in quanti:
    outliers2 = pd.DataFrame(outliers[quanti.columns.get_loc(column)])
    outliers2.columns = ['valeurs']
    quanti_out = quanti_out[~quanti_out[column].isin(outliers2['valeurs'])]
            
len(quanti) - len(quanti_out)



## Essai en definissant les quantiles
high= 0.9999999
low= 0
quantiles_quanti = quanti.quantile([low, high])

quant_out2 = quanti.apply(lambda x: x[(x < quantiles_quanti.loc[high,x.name])], axis=0)

quant_out2.dropna(inplace=True)








# Verification
pd.DataFrame.boxplot(quant_out2, return_type='both')



### Transformation monotone des variables



base_quanti2['MT_OPERATION_DEPOT_3m'] = np.log(base_quanti2['MT_OPERATION_DEPOT_3m']+ 1)
base_quanti2['mt_paiement_chq_3m'] = np.log(-base_quanti2['mt_paiement_chq_3m']+ 1)
base_quanti2['mt_paiement_carte_3m'] = np.log(-base_quanti2['mt_paiement_carte_3m']+ 1)
base_quanti2['REVENU_EST_MM'] = np.log(base_quanti2['REVENU_EST_MM']+ 1)
base_quanti2['SURFACE_FINANCIERE'] = np.log(base_quanti2['SURFACE_FINANCIERE'] -min(base_quanti2['SURFACE_FINANCIERE']) + 1)
base_quanti2['ENCOURS_DAV'] = np.log(base_quanti2['ENCOURS_DAV']+ 1)
base_quanti2['SMS_recus_3m'] = np.sqrt(base_quanti2['SMS_recus_3m'])
base_quanti2['Agence_3m'] = np.sqrt(base_quanti2['Agence_3m'])
base_quanti2['Agence_vente_3m'] = np.sqrt(base_quanti2['Agence_vente_3m'])
base_quanti2['Agence_retrait_3m'] = np.sqrt(base_quanti2['Agence_retrait_3m'])
base_quanti2['Agence_depot_3m'] = np.sqrt(base_quanti2['Agence_depot_3m'])
base_quanti2['Agence_rdv_3m'] = np.sqrt(base_quanti2['Agence_rdv_3m'])
base_quanti2['Agence_vir_3m'] = np.sqrt(base_quanti2['Agence_vir_3m'])
base_quanti2['Connexion_CAEL_3m'] = np.log(base_quanti2['Connexion_CAEL_3m']+ 1)
base_quanti2['Connexion_MaBanque_3m'] = np.log(base_quanti2['Connexion_MaBanque_3m']+ 1)
base_quanti2['Lecture_mess_3m'] = np.sqrt(base_quanti2['Lecture_mess_3m'])
base_quanti2['Ecriture_mess_3m'] = np.sqrt(base_quanti2['Ecriture_mess_3m'])
base_quanti2['Consult_CAtitre_3m'] = np.sqrt(base_quanti2['Consult_CAtitre_3m'])
base_quanti2['Consult_credit_3m'] = np.log(base_quanti2['Consult_credit_3m']+1)
base_quanti2['Consult_epargne_3m'] = np.log(base_quanti2['Consult_epargne_3m']+1)
base_quanti2['Consult_epargne_3m'] = np.log(base_quanti2['Consult_epargne_3m']+1)
base_quanti2['Consult_PACIFICA_3m'] = np.sqrt(base_quanti2['Consult_PACIFICA_3m'])
base_quanti2['Consult_PREDICA_3m'] = np.sqrt(base_quanti2['Consult_PREDICA_3m'])
base_quanti2['Vir_BAM_3m'] = np.sqrt(base_quanti2['Vir_BAM_3m'])
