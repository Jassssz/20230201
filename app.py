import numpy as np
from flask import Flask, request, render_template
import pickle
from pydantic import BaseModel
from flask_pydantic import validate
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)

with open('models/regressor1.pkl','rb') as f:
    model = pickle.load(f)

class QueryParams(BaseModel):
    CleaningNo: float
    Total: float
    CycleNo: float
    AvgFlowrate: float
    AvgFeedCond: float
    AvgCharCur: float
    AvgDisCur: float

@app.route('/apipredict', methods=['POST','GET'])
@validate()
def mcdi_endpoint(body:QueryParams):
    df = pd.DataFrame([body.dict().values()], columns=body.dict().keys())
    pred = model.predict(df)
    return {"Cycle Time": int(pred),'faf': int(body.dict()['CycleNo'])}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    testing = pd.DataFrame()
    for i in range(1,int(int_features[2])+1):
        testing = testing.append({'CleaningNo': int_features[0],
                                'Total': int_features[1]+i,
                                'CycleNo': i,
                                'AvgFlowrate':int_features[3],
                                'AvgFeedCond':int_features[4],
                                'AvgCharCur':int_features[5],
                                'AvgDisCur':int_features[6]}, ignore_index=True)
    # features = [np.array(int_features)]
    # prediction = model.predict(features)
    prediction = model.predict(testing)
    prediction = np.around(prediction,decimals=3)

    img = BytesIO()
    plt.plot(prediction, marker='o', linestyle='--')
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    # plt.savefig('plot.png')
    
    
    # return render_template('index.html', 
    # prediction_text = "Cycle Time: {}".format(int(prediction)))
    # return render_template('index.html', prediction_text = str(prediction[int(int_features[2])-1]), src='my_plot.png')
    return render_template('index.html', prediction_text = str(prediction), plot_url=plot_url)

if __name__ == "__main__":
    app.run()