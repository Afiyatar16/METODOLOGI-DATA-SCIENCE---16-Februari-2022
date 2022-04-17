#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[14]:


get_ipython().system('pip install pandas')
get_ipython().system('pip install seaborn')
get_ipython().system('pip install matplotlib')


# In[16]:


get_ipython().system('pip install sklearn')


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix


# # Import Dataset Serangan Jantung

# In[3]:


df = pd.read_csv ('Dataset Serangan Jantung.csv')
df.head()


# In[5]:


df.dtypes


# In[6]:


import warnings
warnings.filterwarnings("ignore")


# In[7]:


df.describe


# In[8]:


df.mean()


# # Preprocessing

# In[9]:


Var_x = df.drop('Keluaran', axis=1).copy()
Var_x = Var_x.drop('Gender', axis=1)
Var_x = Var_x.drop('cp', axis=1)
Var_x = Var_x.drop('trtbps', axis=1)
Var_x = Var_x.drop('chol', axis=1)
Var_x = Var_x.drop('fbs', axis=1)
Var_x = Var_x.drop('thalachh', axis=1)
Var_x = Var_x.drop('exng', axis=1)
Var_x = Var_x.drop('oldpeak', axis=1)
Var_x = Var_x.drop('slp', axis=1)
Var_x = Var_x.drop('caa', axis=1)
Var_x = Var_x.drop('thall', axis=1)
Var_x = Var_x.drop('Jumlah Pembuluh Darah Utama', axis=1)
Var_x = Var_x.drop('output', axis=1)
Var_x = Var_x.drop('Age', axis=1)
Var_x = Var_x.drop('restecg', axis=1)
Var_x = Var_x.drop('elektrokardiografi istirahat', axis=1)
Var_x


# In[10]:


Var_x.describe()


# In[11]:


Var_x.dtypes


# In[12]:


print("Atribut Umur                     : ",Var_x['Umur'].unique())
print("Atribut Jenis Kelanin            : ",Var_x['Jenis Kelamin'].unique())
print("Atribut Jenis Nyeri Dada         : ",Var_x['Jenis nyeri dada'].unique())
print("Atribut Tekanan Darah            : ",Var_x['Tekanan Darah'].unique())
print("Atribut Kolesetrol               : ",Var_x['Kolestrol'].unique())
print("Atribut Gula Darah Puasa > 120 mg: ",Var_x['Gula Darah Puasa > 120 mg'].unique())
print("Atribut Detak Jantung Maksimum   : ",Var_x['Detak jantung maksimum'].unique())
print("Atribut Angina Akibat Olahraga   : ",Var_x['Angina akibat olahraga'].unique())
print("Atribut Kemiringan Segment ST    : ",Var_x['Kemiringan segmen ST'].unique())


# In[13]:


Var_x.to_csv("Variabel_X.csv", index=False)


# # Melakukan Encoder untuk membuat data yang bersifat objek menjadi data interger agar dapat dihitung dengan menggunakan model prediksi

# In[14]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
Var_x['Umur']=enc.fit_transform(Var_x['Umur'].values)
Var_x['Jenis Kelamin']=enc.fit_transform(Var_x['Jenis Kelamin'].values)
Var_x['Jenis nyeri dada']=enc.fit_transform(Var_x['Jenis nyeri dada'].values)
Var_x['Tekanan Darah']=enc.fit_transform(Var_x['Tekanan Darah'].values)
Var_x['Kolestrol']=enc.fit_transform(Var_x['Kolestrol'].values)
Var_x['Gula Darah Puasa > 120 mg']=enc.fit_transform(Var_x['Gula Darah Puasa > 120 mg'].values)
Var_x['Detak jantung maksimum']=enc.fit_transform(Var_x['Detak jantung maksimum'].values)
Var_x['Angina akibat olahraga']=enc.fit_transform(Var_x['Angina akibat olahraga'].values)
Var_x['Kemiringan segmen ST']=enc.fit_transform(Var_x['Kemiringan segmen ST'].values)
Var_x


# In[15]:


print("Atribut Umur                     : ",Var_x['Umur'].unique())
print("Atribut Jenis Kelanin            : ",Var_x['Jenis Kelamin'].unique())
print("Atribut Jenis Nyeri Dada         : ",Var_x['Jenis nyeri dada'].unique())
print("Atribut Tekanan Darah            : ",Var_x['Tekanan Darah'].unique())
print("Atribut Kolesetrol               : ",Var_x['Kolestrol'].unique())
print("Atribut Gula Darah Puasa > 120 mg: ",Var_x['Gula Darah Puasa > 120 mg'].unique())
print("Atribut Detak Jantung Maksimum   : ",Var_x['Detak jantung maksimum'].unique())
print("Atribut Angina Akibat Olahraga   : ",Var_x['Angina akibat olahraga'].unique())
print("Atribut Kemiringan Segment ST    : ",Var_x['Kemiringan segmen ST'].unique())


# In[16]:


len(Var_x)


# In[17]:


Var_y = df['Keluaran'].copy()
Var_y


# In[ ]:





# ####
# Keterangan:
# 
# umur
# 25-45 : Dewasa = 0
# 46-64 : Lansia = 1
# =>65  : Manula = 2
# 
# jenis kelamin:
# laki-laki = 0 
# perempuan = 1
# 
# cp: jenis nyeri dada (1 = angina tipikal; 2 = Nyeri non-Angina; 3 =
# Asimtomatik; 0 = Angina Atipikal)
# 
# trtbps: tekanan darah istirahat (dalam mm Hg saat masuk rumah sakit)
# 
# chol: kolesterol serum dalam mg/dl
# 
# fbs: gula darah puasa > 120 mg/dl (1 = benar; 0 = salah)
# 
# restecg: hasil elektrokardiografi istirahat (1 = normal; 2 = memiliki
# kelainan gelombang ST-T ; 0 = hipertrofi)
# 
# thalachh: detak jantung maksimum tercapai
# 
# exng: angina akibat olahraga (1 = ya; 0 = tidak)
# 
# oldpeak: depresi ST yang diinduksi oleh olahraga relatif terhadap istirahat
# 
# slp: kemiringan segmen ST latihan puncak (2 = upsloping; 1 = flat;
# 0 = downsloping)
# 
# caa: jumlah pembuluh darah utama (0-3) yang diwarnai dengan fluoroskopi
# 
# tinggi: 2 = biasa; 1 = cacat tetap; 3 = cacat reversibel
# 
# output: 0= lebih sedikit kemungkinan pelekatan jantung 1= lebih banyak perubahan serangan jantung.

# In[ ]:





# # Model Klasifikasi menggunakan Decision Tree

# In[18]:


random = 130
xtrain,xtest,ytrain,ytest = train_test_split(Var_x,Var_y, test_size=0.8, random_state = random)
tree_data = DecisionTreeClassifier(random_state=random)
train_data = tree_data.fit(Var_x,Var_y)
train_data


# In[19]:


print(ytest.value_counts(normalize=True))


# In[20]:


print('Nilai Akurasi Training: ', tree_data.score(xtest,ytest))
print('Nilai Akurasi: ', tree_data.score(xtrain,ytrain))


# In[21]:


print('Data yang ditrainig sebanyak 20% = ', len(train_data.predict(xtrain)), 'data')
print('Data yang ditesting sebanyak 80% = ', len(train_data.predict(xtest)), 'data')


# In[22]:


print('Data yang ditrainig sebanyak 20%')
train_data.predict(xtrain)


# In[23]:


print('Data yang ditesting sebanyak 80%')
train_data.predict(xtest)


# In[24]:


from sklearn import metrics
from sklearn.metrics import accuracy_score
y_pred = train_data.predict(xtest)

print("---------------------------------------------------------")
print('Akurasi Decision Tree (Training): {0:0.4f}'.format(metrics.accuracy_score(ytest,y_pred)*100), '%')
#performa
from sklearn.metrics import classification_report
y_perform = train_data.predict(xtest)
print("---------------------------------------------------------")
print(classification_report(ytest,y_perform))
print("---------------------------------------------------------")


# In[25]:


plt.figure(figsize=(15, 7.5))
plot_tree(train_data, filled=True, rounded=True, class_names=['Tidak', 'Ya'], feature_names = Var_x.columns)


# In[26]:


akurasi = ((107+124)/(107+124+1+11))*100
precision = ((107)/(107+1))*100
recal = ((107)/(107+11))*100
f1 = ((2*recal*precision)/(recal+precision))
print("Akurasi pasien diprediksi terkena serangan Jantung       : {0:0.4f} " .format(akurasi),'%')
print("Tingkat keakuratan data yang diminta dengan hasil        : {0:0.4f} " .format(precision),'%')
print("Tingkat keberhasilan model menemukan sebuah informasi    : {0:0.4f} " .format(recal),'%')
print("Rata-rata precision dan recall                           : {0:0.4f} " .format(f1),'%')


# In[27]:


get_ipython().system('pip install plotly')


# In[28]:


from sklearn import tree
max_depths = np.array([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])

train_score = []
test_score = []
for d in max_depths:
    dtc = tree.DecisionTreeClassifier(max_depth=d)
    dtc.fit(xtrain, ytrain)
    train_score.append(dtc.score(xtrain, ytrain))
    test_score.append(dtc.score(xtest, ytest))

import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(x=max_depths, y=train_score,
                    mode='lines+markers',
                    name='train'))
fig.add_trace(go.Scatter(x=max_depths, y=test_score,
                    mode='lines+markers',
                    name='test'))

fig.update_layout(
    xaxis_title="max_depths",
    yaxis_title="score",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()


# In[29]:


import warnings 
warnings.filterwarnings("ignore")
                          #Umur	Jenis Kelamin	Jenis nyeri dada	Tekanan Darah	Kolestrol	Gula Darah Puasa > 120 mg	Detak jantung maksimum	Angina akibat olahraga	Kemiringan segmen ST
Prediksc = train_data.predict([[0,	0,	2,	2,	2,	0,	1,	0,	0]])
Prediksc


# In[ ]:


print("===============================================================================================================================================================")
print("                                                          Program Pengecakan Potensi Serangan Jantung                 ")
print("                                                                      Simple Cek                                      ")
print("===============================================================================================================================================================")
print("Keterangan:                                                                  ")
print("Umur: 25-45              : 25-45 = Dewasa = (0), 46-64 = Lansia = (1), =>65 : Manula = (2)          ")
print("Jenis Kelamin            : Laki-laki = (1), Perempuan = (0)                              ")
print("Jenis Nyeri Dada         : (0) = angina atipikal, (1) = angina tipikal, (2) = nyeri non- angina, (3) = asimtomatik")
print("Tekanan Darah            : Hipertensi T. I = >=140 - 159 : (0), Hipertensi T. II = >=160 : (1), Normal = >= 90 - 120: (2), PraHipertensi = >=130-139 : (3)")
print("Kolesterol               : Normal = >=125 - 199 : (1), Batas Tinggi = >=200 - 239 : (0), Tinggi = >=240 - 565 : (2)")
print("Gula Darah > 120 mg      : Ya = (1), Tidak = (0)")
print("Detak Jantung Max.       : Normal = >= 110 - 149 : (1), Tinggi = >= 150 : (2), Kurang = >= 50 - 109 : (0)")
print("Angina Akibat Olahraga   : Ya = (1), Tidak = (0)")
print("Kemiringan Segment ST    : Datar = (0), Menurun = (1), Miring = (2)")
print()
print("---------------------------------------------------------------------------------------------------------------------------------------------------------------")
print()
nama     = input('Inputkan nama Anda                : ')
umur     = input('Inputkan Umur Anda                : ')

print           ('===========================================')
print           ('Ketik 0 jika Laki-laki, ketik 1 jika Perempuan')
jk       = input('Inputkan Jenis Kelamin Anda       : ')
print           ('===========================================')
jn       = input('Inputkan Jenis Nyeri              : ')
td       = input('Inputkan Tekanan Darah            : ')
kole     = input('Inputkan Kolesterol               : ')
gd       = input('Inputkan Gula Darah > mg          : ')
dj       = input('Inputkan Detak Jantung Maksimum   : ')
angina   = input('Inputkan Angina Akibat Olahraga   : ')
kemst    = input('Inputkan Kemiringan Segmen ST     : ')
print()
Prediksi = train_data.predict([[umur,jk,jn,td,kole,gd,dj,angina,kemst]])
print("===============================================================================================================================================================")
print('Hasil dari prediksi adalah',Prediksi, 'terhadap serangan jantung')
print("===============================================================================================================================================================")
print("Jika Hasil Prediksi Adalah ya Segera Periksa ke Dokter Spesialis Untuk Penanganan Lebih Lanjut")
print("Catatan:")
print("1. Beritahu gejala-gejala yang dialami")
print("2. Jaga kolesterol dalam tubuh")
print("3. Jaga tekanan darah")
print("===============================================================================================================================================================")


# In[30]:


Var_y.to_csv("Variabel_y.csv", index=False)


# In[ ]:





# In[31]:


import pickle


# In[34]:


with open('dtree_pickle','wb') as r:
    pickle.dump(train_data,r)


# In[36]:


with open('dtree_pickle','rb') as r:
    dtreep = pickle.load(r)


# In[43]:


dtreep.predict([[0,	0,	2,	2,	2,	0,	1,	0,	0]])

