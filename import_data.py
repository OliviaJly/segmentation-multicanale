# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 14:08:12 2017

@author: Lucie
"""

##  Copier le code dans la console python




############ IMPORT DES DONNEES : 

# Utilisation de la librairie Pandas (plus facile pour travailler avec des dataframes) :
#import pandas as pd


# Definition du chemin où sont situées les données :
path = 'C:/Users/Richard/Documents/GitHub/Segmentation-multicanale/Données'  

# Import des données
df = pd.read_csv(path + '/OLIVIA_BASE_PART_092016.csv',nrows=20000,delimiter=";",encoding = "ISO-8859-1",
                 dtype={"IDPART_CALCULE":object,"IDCLI_CALCULE":object})
# encoding = "ISO-8859-1" est nécessaire, le défaut ne fonctionne pas (impossible de lire les données en utf-8,
# je ne sais pas trop pourquoi
# dtype change le type de variables, object = charactère
 







############ METTRE LES DONNEES SOUS LE BON FORMAT :

# Changement du format de date_part en date:
df["date_part"] = pd.to_datetime(df["date_part"])

# Verification du type des données : 
types = df.dtypes  # Ok. (Float : int avec décimales)






############ MMISE EN FORME DE LA BASE D'ETUDE (ACP) :

# Selection des variables quanti + id part dans une nouvelle table df_quanti :
df_quanti = pd.concat([df['IDPART_CALCULE'],df.select_dtypes(include=['float64','int64'])],axis=1)
# On garde seulement les variables quanti (float et int), et on fait un cbind (axis=1 veut dire qu'on concatene
# sur les colonnes

###### Selection des variables interessantes :

    
### Code suppression des variables sur 1 et 2 mois :
    
var_names = list(df_quanti.columns.values)   # Liste contenant les noms des variables
sub_1m = '_1m' # Pattern à supprimer
sub_2m = '_2m' # Pattern à supprimer
        
x = list(range(0,len(var_names)))        # Liste qui contiendra des booleens
for i in range(0,len(var_names)):
    if sub_1m in var_names[i] or sub_2m in var_names[i]:
        x[i] = False
    else:
        x[i] = True
# x = True quand le nom de la variable ne contient aucun des pattern. On gardera donc les variables quand x = True

var = pd.DataFrame({'vf': x,'noms': var_names}) # Creation d'un data frame intermédiaire
var = var[var.vf == False] # On garde seulement les variables quand c'est False parce que panda n'a pas de
# fonction keep, mais uniquement de fonction drop

var_drop = list(var['noms']) # Extraction de la liste des variables dont on veut se debarasser 
base_quanti = df_quanti.drop(var_drop,axis=1)   # Drop les variables qu'on ne veut plus
### Fin code suppression des variables sur 1 et 2 mois


### Recoder les variables comme dans le code SAS (transformations monotones)

# Utilisation du package numpy :
import numpy as np

base_quanti['MT_OPERATION_DEPOT_3m'] = np.log(base_quanti['MT_OPERATION_DEPOT_3m']+ 1)
base_quanti['mt_paiement_chq_3m'] = np.log(-base_quanti['mt_paiement_chq_3m']+ 1)
base_quanti['mt_paiement_carte_3m'] = np.log(-base_quanti['mt_paiement_carte_3m']+ 1)
base_quanti['REVENU_EST_MM'] = np.log(base_quanti['REVENU_EST_MM']+ 1)
base_quanti['DEPENSES_RECURRENTES_EST_M'] = np.log(base_quanti['DEPENSES_RECURRENTES_EST_M']+ 1)
base_quanti['SURFACE_FINANCIERE'] = np.log(base_quanti['SURFACE_FINANCIERE'] -min(base_quanti['SURFACE_FINANCIERE']) + 1)
base_quanti['ENCOURS_DAV'] = np.log(base_quanti['ENCOURS_DAV']+ 1)
base_quanti['SMS_recus_3m'] = np.sqrt(base_quanti['SMS_recus_3m'])
base_quanti['SMS_authentif_3m'] = np.sqrt(base_quanti['SMS_authentif_3m'])
base_quanti['Agence_3m'] = np.sqrt(base_quanti['Agence_3m'])
base_quanti['Agence_vente_3m'] = np.sqrt(base_quanti['Agence_vente_3m'])
base_quanti['Agence_retrait_3m'] = np.sqrt(base_quanti['Agence_retrait_3m'])
base_quanti['Agence_depot_3m'] = np.sqrt(base_quanti['Agence_depot_3m'])
base_quanti['Agence_rdv_3m'] = np.sqrt(base_quanti['Agence_rdv_3m'])
base_quanti['Agence_vir_3m'] = np.sqrt(base_quanti['Agence_vir_3m'])
base_quanti['Connexion_CAEL_3m'] = np.log(base_quanti['Connexion_CAEL_3m']+ 1)
base_quanti['Connexion_MaBanque_3m'] = np.log(base_quanti['Connexion_MaBanque_3m']+ 1)
base_quanti['Duree_CAEL_3m'] = np.log(base_quanti['Duree_CAEL_3m']+ 1)
base_quanti['Duree_MaBanque_3m'] = np.log(base_quanti['Duree_MaBanque_3m']+ 1)
base_quanti['Actions_CAEL_3m'] = np.log(base_quanti['Actions_CAEL_3m']+ 1)
base_quanti['Actions_MaBanque_3m'] = np.log(base_quanti['Actions_MaBanque_3m']+ 1)
base_quanti['Lecture_mess_3m'] = np.sqrt(base_quanti['Lecture_mess_3m'])
base_quanti['Ecriture_mess_3m'] = np.sqrt(base_quanti['Ecriture_mess_3m'])
base_quanti['Ges_Alerte_CAEL_3m'] = np.sqrt(base_quanti['Ges_Alerte_CAEL_3m'])
base_quanti['Consult_BGPI_3m'] = np.sqrt(base_quanti['Consult_BGPI_3m'])
base_quanti['Consult_carte_3m'] = np.sqrt(base_quanti['Consult_carte_3m'])
base_quanti['Consult_CAtitre_3m'] = np.sqrt(base_quanti['Consult_CAtitre_3m'])
base_quanti['Consult_credit_3m'] = np.log(base_quanti['Consult_credit_3m']+1)
base_quanti['Consult_epargne_3m'] = np.log(base_quanti['Consult_epargne_3m']+1)
base_quanti['Consult_epargne_3m'] = np.log(base_quanti['Consult_epargne_3m']+1)
base_quanti['Dem_contact_3m'] = np.sqrt(base_quanti['Dem_contact_3m'])
base_quanti['Consult_PACIFICA_3m'] = np.sqrt(base_quanti['Consult_PACIFICA_3m'])
base_quanti['Consult_PREDICA_3m'] = np.sqrt(base_quanti['Consult_PREDICA_3m'])
base_quanti['Simul_creditconso_3m'] = np.sqrt(base_quanti['Simul_creditconso_3m'])
base_quanti['Simul_credithabitat_3m'] = np.sqrt(base_quanti['Simul_credithabitat_3m'])
base_quanti['Consult_Sofinco_3m'] = np.sqrt(base_quanti['Consult_Sofinco_3m'])
base_quanti['Vir_BAM_3m'] = np.sqrt(base_quanti['Vir_BAM_3m'])



# Retrait des variables correlees et non disciminantes
base_quanti2 = base_quanti.drop(['DEPENSES_RECURRENTES_EST_M','SMS_recus_3m','SMS_authentif_3m'
                                ,'Automate_depot_3m','Automate_retrait_3m','Duree_CAEL_3m','Duree_CAEL_3m',
                                'Actions_CAEL_3m','Actions_MaBanque_3m','Ges_Alerte_CAEL_3m','Consult_BGPI_3m',
                                'Consult_carte_3m','Dem_contact_3m','Simul_creditconso_3m',
                                'Simul_credithabitat_3m','Consult_Sofinco_3m','Email_3m','Appels_entrants_3m',
                                'Duree_MaBanque_3m'],axis=1)
# base_quanti2 est la même base d'étude que pour l'ACP sous SAS


# Enregistre la base en csv :
base_quanti2.to_csv(path + '/quanti_trans.csv',index=False)

