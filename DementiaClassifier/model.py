# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

df_long = pd.read_csv('Data_OASIS3.csv')

df_clean= df.dropna()
df_clean[df_clean["Diagnosis"] == "AD Dementia"]
d = {'male': 1, 'female': 0}
df_clean['Gender'].replace(d,inplace = True)

#Normalisation des données de diagnostic
d1 = {'AD dem Language dysf after': 'AD Dementia',
      'AD dem Language dysf prior' : 'AD Dementia',  
      'AD dem Language dysf with': 'AD Dementia' ,
      'AD dem cannot be primary' : 'AD Dementia', 
      'AD dem distrubed social- after' : 'AD Dementia',
      'AD dem distrubed social- prior' : 'AD Dementia', 
      'AD dem distrubed social- with' : 'AD Dementia',
      'AD dem visuospatial- prior' : 'AD Dementia', 
      'AD dem visuospatial- with' : 'AD Dementia',
      'AD dem w/CVD contribut' : 'AD Dementia', 
      'AD dem w/CVD not contrib' : 'AD Dementia',
      'AD dem w/Frontal lobe/demt at onset':'AD Dementia',
      'AD dem w/PDI after AD dem contribut' : 'AD Dementia',
      'AD dem w/PDI after AD dem not contrib':'AD Dementia', 
      'AD dem w/depresss  contribut':'AD Dementia',
      'AD dem w/depresss  not contribut':'AD Dementia',
      'AD dem w/depresss- contribut':'AD Dementia', 
      'AD dem w/depresss- not contribut':'AD Dementia', 
      'AD dem w/oth (list B) contribut':'AD Dementia',
      'AD dem w/oth (list B) not contrib':'AD Dementia', 
      'AD dem w/oth unusual feat/subs demt' : 'AD Dementia', 
      'AD dem w/oth unusual features/demt on' : 'AD Dementia',
      'AD dem/FLD prior to AD dem' : 'AD Dementia', 
      'DAT':'AD Dementia',
      'DAT w/depresss not contribut' : 'AD Dementia',
      'DLBD- primary' : 'DLBD',
      'DLBD- secondary' :'DLBD',
      'Dementia/PD- primary' : 'Parkinson Dementia',
      'Frontotemporal demt. prim':'Frontotemporal dementia',
      'Incipient Non-AD dem':'Incipient dementia',
      'Incipient demt PTP':'Incipient dementia',
      'No dementia':'Cognitively normal',
      'Non AD dem- Other primary': 'Incipient dementia',
      'ProAph w/o dement' : 'Dementia', 
      'Unc: impair reversible' : 'Uncertain Dementia',
      'Unc: ques. Impairment' : 'Uncertain Dementia',     
      'Vascular Demt  primary': 'Vascular Dementia',              
      'Vascular Demt- primary' : 'Vascular Dementia',           
      'Vascular Demt- secondary': 'Vascular Dementia', 
      'uncertain  possible NON AD dem':'Uncertain Dementia',     
      'uncertain dementia':'Uncertain Dementia' ,               
      'uncertain- possible NON AD dem':'Uncertain Dementia',
      '0.5 in memory only':'Dementia'    
     }
df_clean['Diagnosis'].replace(d1,inplace = True)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

X= df_clean.loc[:, df_clean.columns != 'Diagnosis']
y= df_clean.iloc[:, 5]

#We perform target encoding
from sklearn.preprocessing import LabelEncoder 
ly = LabelEncoder()
y = ly.fit_transform(y)

import numpy as np
#from imblearn.datasets import fetch_datasets
from kmeans_smote import KMeansSMOTE


[print('Class {} has {} instances'.format(label, count))
 for label, count in zip(*np.unique(y, return_counts=True))]

kmeans_smote = KMeansSMOTE(
    sampling_strategy= 'minority',
    kmeans_args={
        'n_clusters': 100
    },
    smote_args={
        'k_neighbors': 10
    }
)
X_resampled, y_resampled = kmeans_smote.fit_sample(X, y)

[print('Class {} has {} instances after oversampling'.format(label, count))
 for label, count in zip(*np.unique(y_resampled, return_counts=True))]
 
 

#Splitting Training and Test Set
#Since we have a very small dataset, we will train our model with all availabe data.
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X_resampled,y_resampled,test_size=0.2)

from sklearn.svm import SVC
svc1 = SVC(C=50,kernel='rbf',gamma=1)     
svc1.fit(x_train,y_train)



# Saving model to disk
pickle.dump(rf_model_opt_2, open('svc1.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
