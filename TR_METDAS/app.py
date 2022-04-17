#!/usr/bin/env python
# coding: utf-8

# # Import Library

# In[2]:


from flask import Flask, render_template, request, redirect, jsonify
import pickle
import sklearn
import numpy as np                        
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':

        dtreepred = pickle.load(open('dtree_pickle','rb'))

        nama = str(request.form['nama'])
        umur = float(request.form['umur'])
        jenis_kelamin = float(request.form['jk'])
        jenis_nyeri  = float(request.form['jn'])
        tekanan_darah = float(request.form['td'])
        kolesterol = float(request.form['kole'])
        gula_darah = float(request.form['gd'])
        detak_jantung = float(request.form['dj'])
        anginast = float(request.form['angina'])
        kemiringan = float(request.form['kemset'])

        datas = np.array([umur,jenis_kelamin,jenis_nyeri,tekanan_darah,kolesterol,gula_darah,detak_jantung,anginast,kemiringan])
        datas = np.reshape(datas, (1, -1))

        SeranganJantung = dtreepred.predict(datas)

        return render_template('hasil.html', finalData=SeranganJantung, umur=umur, nama=nama)
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

