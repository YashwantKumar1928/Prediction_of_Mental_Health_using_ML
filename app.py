from flask import Flask,request,render_template

import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,MinMaxScaler


model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('form.html')


# Assuming the model was trained with label encoded features
gender_encoder = LabelEncoder()
family_history_encoder = LabelEncoder()
benefits_encoder = LabelEncoder()
care_options_encoder = LabelEncoder()
anonymity_encoder = LabelEncoder()
leave_encoder = LabelEncoder()
work_interfere_encoder = LabelEncoder()

# Fit the encoders on the training data and save them
# For demonstration, let's assume we already have the encoders fitted and saved
# You can load them similar to the model loading step
gender_encoder.classes_ = np.array(['female', 'male', 'trans'])
family_history_encoder.classes_ = np.array(['no', 'yes'])
benefits_encoder.classes_ = np.array(["don't know",'no','yes'])
care_options_encoder.classes_ = np.array(['no', 'not sure','yes'])
anonymity_encoder.classes_ = np.array(["don't know",'no','yes'])
leave_encoder.classes_ = np.array(["Don't know", 'Somewhat difficult', 'Somewhat easy', 'Very difficult', 'Very easy'])
work_interfere_encoder.classes_ = np.array(["Don't know", 'Never', 'Often', 'Rarely', 'Sometimes'])

@app.route('/predict',methods=['GET','POST'])
def submit():
    if request.method=='POST':
        age = request.form['age']
        gender = request.form['gender']
        family_history = request.form['family_history']
        benefits = request.form['benefits']
        care_options = request.form['care_options']
        anonymity = request.form['anonymity']
        leave = request.form['leave']
        work_interfere = request.form['work_interfere']
        #Scale age
        scaler = MinMaxScaler()
        age = float(age)
        age_scaled = scaler.fit_transform([[age]])
    # Encode form data
        data = np.array([[
            age_scaled[0][0],
            gender_encoder.transform([gender])[0],
            family_history_encoder.transform([family_history])[0],
            benefits_encoder.transform([benefits])[0],
            care_options_encoder.transform([care_options])[0],
            anonymity_encoder.transform([anonymity])[0],
            leave_encoder.transform([leave])[0],
            work_interfere_encoder.transform([work_interfere])[0]
        ]])

    # Make a prediction
        prediction = model.predict(data)

    # Process the prediction result if necessary
        result = prediction[0]  # Assuming a single prediction
        if result==0:
            output=f"Predicted class is:{result} . No need Mental consultation"
        else:
            output=f"Predicted class is:{result} . Need Mental consultation"


    # Render a result page or return the result as JSON
        return render_template('form.html', result=output)

if __name__ == '__main__':
	app.run(debug=True)
