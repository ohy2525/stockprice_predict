import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

##학습데이터, 정답데이터 -> 월~목, 1set
# LSTM -> Numpy 배열
def Scaler_to_np(df):
    df_list = []
    df = np.array(df)
    for i in range(0, len(df) - 3, 5):
        df_s = df[i : i + 4]
        scaler = StandardScaler()
        df_scaler = scaler.fit_transform(df_s)
        df_list.append(df_scaler)
        
    return np.array(df_list) 

def data_split(data):
    pred_x = data[['weekday','High','Low','Open','Close','Close_ratio','Body','Volume','5MA', '10MA',
        '20MA', '60MA', '120MA', 'cross']]
    pred_y = data['Up']

    return pred_x, pred_y

def make_test_data(data):
    test_data = data[-30:]  #2월부터 3월까지 현재 6주 데이터가 있어서 30개

    return test_data

def load_data(data_path_pattern):
    data_path = glob.glob(data_path_pattern)
    df = pd.read_csv(data_path[0],index_col = 'Date')

    return df


RETURN_SUCCESS = 0
RETURN_FAILURE = -1
# Inpou Model Directory
INPUT_MODEL_PATH = "./model/model.h5"
PREPROCESSED_DATA_PATTERN = "./preprocessed_data/*"
INPUT_MODEL_PATH = "./model/model.h5"


def main():
    print("==================================================================")
    print("LSTM모델 학습")
    print("==================================================================")

    
    #원본 데이터 로드
    df = load_data(PREPROCESSED_DATA_PATTERN)

    test_data = make_test_data(df)

    pred_x, pred_y = data_split(test_data)

    pred_x_array = Scaler_to_np(pred_x)
    print(pred_x_array.shape)
    pred_y_new = pred_y[4::5]
    print(len(pred_y_new))

    # 모델 파일 읽기
    if not os.path.exists(INPUT_MODEL_PATH):
        print('모델 파일이 없습니다.')
        return RETURN_FAILURE
    model = keras.models.load_model(INPUT_MODEL_PATH)

    pred = model.predict(pred_x_array)
    pred = np.where(pred < 0.5, 0, 1)
    print("===========================================================")
    print(f"accuracy={accuracy_score(y_true=pred_y_new,y_pred = pred)}")

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()