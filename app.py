import numpy as np
import pickle
from datapreprocessor import *
from flask import Flask
from flask import render_template,request
import pandas as pd

app=Flask(__name__)

preprocessor=pickle.load(open('scaling.pkl','rb'))
feature_selection=pickle.load(open('feature_selection.pkl','rb'))
model=pickle.load(open('xgb.pkl','rb'))



@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def get_data():

    vendorid=float(request.form.get('vendor'))
    trip_distance=float(request.form.get('trip_distance'))
    ratecode_ID=float(request.form.get('ratecode'))
    payment_type=(request.form.get('payment_type'))
    extra=float(request.form.get('extra'))
    tip_amount=float(request.form.get('tip_amount'))
    tolls_amount=float(request.form.get('tolls_amount'))
    Improvement_surcharge=float(request.form.get('Improvement_surcharge'))
    Congestion_surcharge=float(request.form.get('Congestion_surcharge'))
    Airport_fee=float(request.form.get('Airport_fee'))
    Day=float(request.form.get('Day'))
    Hour=float(request.form.get('Hour'))
    duration_in_min=float(request.form.get('duration_in_min'))
    
    column=['VendorID','trip_distance','RatecodeID','payment_type','extra','tip_amount','tolls_amount',
                     'improvement_surcharge','congestion_surcharge','Airport_fee','Day','Hour','duration_in_min']
    data=np.array([float(x) if i!=3 else x for i,x in enumerate(request.form.values())])
    data[1]=np.log1p(float(data[1]))
    data[2]=np.log1p(float(data[2]))
    data[5]=(np.abs(float(data[5]))**(1/3))
    data[12]=(np.abs(float(data[12]))**(1/3))
    data=data.reshape(1,13)
    data=pd.DataFrame(data=data,columns=column)


    new_data=preprocessor.transform(data)
    new_data=feature_selection.transform(new_data)
    final_output=model.predict(new_data)
    return render_template('home.html',prediction="The predicted taxi fare is : {}".format(final_output[0]))

if __name__=='__main__':
    app.run(debug=True)