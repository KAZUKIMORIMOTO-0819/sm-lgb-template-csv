import os
from io import BytesIO, StringIO
import lightgbm as lgb
import json
import logging
import json
import tarfile
import pandas as pd

# ログの設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def model_fn(model_dir):
    logger.info("モデルをロードしています。")
    model_path = os.path.join(model_dir, 'model.txt')
    model = lgb.Booster(model_file=model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        df1 = pd.DataFrame(input_data['data1'])
        df2 = pd.DataFrame(input_data["data2"])

        df = df1.copy()
        df["feature1"]=df1["feature1"] + df2["feature10"]
        return df
        
    elif request_content_type == 'text/csv':
        s = io.StringIO(request_body)
        df = pd.read_csv(s, header=None)
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    output = change_output(prediction)
    if response_content_type == 'application/json':
        return json.dumps({'predictions': output.tolist()})
    elif response_content_type == 'text/csv':
        s = io.StringIO()
        pd.DataFrame(output).to_csv(s, index=False, header=False)
        return s.getvalue()
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

def change_output(prediction, ):
    return prediction*10000