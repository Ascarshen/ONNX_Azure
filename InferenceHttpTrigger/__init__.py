import azure.functions as func
import pandas as pd
import numpy as np
import onnxruntime as rt
import logging
import json



respeck = rt.InferenceSession("./InferenceHttpTrigger/model/respeck.onnx") 
thingy = rt.InferenceSession("./InferenceHttpTrigger/model/thingy.onnx")
input_name_respeck = respeck.get_inputs()[0].name
label_name_respeck = respeck.get_outputs()[0].name
input_name_thingy = thingy.get_inputs()[0].name
label_name_thingy = thingy.get_outputs()[0].name

class_labels = {
    'Desk work': 0,
    'Walking at normal speed': 1,
    'Standing': 2 ,
    'Sitting bent forward': 3,
    'Sitting': 4,
    'Sitting bent backward': 5,
    'Lying down right': 6,
    'Lying down left':7 ,
    'Lying down on back':8 ,
    'Lying down on stomach': 9, 
    'Movement': 10, 
    'Running': 11, 
    'Climbing stairs':12,
    'Descending stairs': 13
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)



def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    sensor_data = req.get_json()

    if not sensor_data:
        try:
            sensor_data = req.get_json()
        except ValueError:
            pass
        else: 
            sensor_data = sensor_data
            # writeCsv(sensor_data.get('respeck_json'),sensor_data.get('thingy_json'))

    if  sensor_data.get('type') == 'respeck':
        result = rep_predict()
        numpyData = {"result": result}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyEncoder)
    
        return func.HttpResponse(encodedNumpyData)

    if  sensor_data.get('type') == 'thingy':
        result = thingy_predict()
        numpyData = {"result": result}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyEncoder)
        return func.HttpResponse(encodedNumpyData)

    if  sensor_data.get('type') == 'both':
        result = thingy_predict()
        numpyData = {"result": result}
        encodedNumpyData = json.dumps(numpyData, cls=NumpyEncoder)
        return func.HttpResponse(encodedNumpyData)   
    
    
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )




def rep_predict():
    resDf = pd.read_csv("./InferenceHttpTrigger/cache/res.csv")
    res_interest = [
        'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z'
    ]

    resFeatures = []
    
    for feature in res_interest:
        data = np.array(resDf[feature].to_list())
        resFeatures.append(np.sum(data))
        resFeatures.append(np.median(data))
        resFeatures.append(np.mean(data))
        resFeatures.append(50)
        resFeatures.append(np.std(data))
        resFeatures.append(np.var(data))
        resFeatures.append(np.sqrt(np.mean(data**2)))
        resFeatures.append(max(data))
        resFeatures.append(max(map(abs, data)))
        resFeatures.append(min(data))
    resFeatures = np.array([resFeatures])
    resFeatures = resFeatures.astype(np.float32)
    pred_onx = respeck.run([label_name_respeck], {input_name_respeck: resFeatures})[0]
    return pred_onx
    
def thingy_predict():
    
    thiDf = pd.read_csv("./InferenceHttpTrigger/cache/thi.csv")

    thi_interest = [
        'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x',
        'mag_y', 'mag_z'
    ]

    thiFeatures = []

    for feature in thi_interest:
        data = np.array(thiDf[feature].to_list())
        thiFeatures.append(np.sum(data))
        thiFeatures.append(np.median(data))
        thiFeatures.append(np.mean(data))
        thiFeatures.append(50)
        thiFeatures.append(np.std(data))
        thiFeatures.append(np.var(data))
        thiFeatures.append(np.sqrt(np.mean(data**2)))
        thiFeatures.append(max(data))
        thiFeatures.append(max(map(abs, data)))
        thiFeatures.append(min(data))
    thiFeatures = np.array([thiFeatures])
    thiFeatures = thiFeatures.astype(np.float32)
    pred_onx = thingy.run([label_name_thingy], {input_name_thingy: thiFeatures})[0]
    return pred_onx

def writeCsv(respeck_json, thingy_json):
    respeck_data = json.loads(respeck_json)
    thingy_data = json.loads(thingy_json)
    file = open("./InferenceHttpTrigger/cache/thi.csv", "w")
    file.write(
        "timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z\n"
    )
    for i in thingy_data:
        file.write(str(i)[1:-1] + "\n")

    file = open("./InferenceHttpTrigger/cache/res.csv", "w")
    file.write("timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z\n")
    for i in respeck_data:
        file.write(str(i)[1:-1] + "\n")
    file.close()
