import os
import sys
import glob

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, SimpleRNN, GRU, LSTM, Dense, GlobalMaxPool1D, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def data_split(data):
    data = data[:-30]  #확인하려는 데이터(2월, 3월)는 학습에서 제외
    df_train = data[: '2020-12-31']
    df_test = data['2021-01-01':]
    X_train = df_train[['weekday','High','Low','Open','Close','Close_ratio','Body','Volume','5MA', '10MA',
        '20MA', '60MA', '120MA', 'cross']]
    y_train = df_train['Up']
    X_test = df_test[['weekday','High','Low','Open','Close','Close_ratio','Body','Volume','5MA', '10MA',
        '20MA', '60MA', '120MA', 'cross']]
    y_test = df_test['Up']

    return(X_train, y_train, X_test, y_test)

##학습데이터, 정답데이터 -> 월~목, 1set
# LSTM -> Numpy 배열
def Scaler_to_np(df):
    df_list = []
    df = np.array(df)
    for i in range(0, len(df) - 3, 4):
        df_s = df[i : i + 4]
        scaler = StandardScaler()
        df_scaler = scaler.fit_transform(df_s)
        df_list.append(df_scaler)
        
    return np.array(df_list) 

#시계열데이터 모델 검증하기    
def sequence_fold(X_TRAIN, Y_TRAIN):
    valid_scores = []
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_indices, value_indices) in enumerate(tscv.split(X_TRAIN)):
        X_train, X_valid = X_TRAIN[train_indices], X_TRAIN[value_indices]
        y_train, y_valid = Y_TRAIN[train_indices], Y_TRAIN[value_indices]
        
        model = gen_lstm_compile(X_train)
        model.fit(X_train, y_train, epochs = 100, batch_size = 64)
        
        y_pred = model.predict(X_valid)
        y_pred = np.where(y_pred < 0.5, 0, 1)
        score = accuracy_score(y_valid, y_pred)
        print(f"fold : {fold} Score : {score}")
        
        valid_scores.append(score)
    
    print(f"valid_scores : {valid_scores}")
    cv_score = np.mean(valid_scores)
    print(f"cv_score:{cv_score}")



#LSTM 모델 생성     
def gen_lstm_compile(df):
    model = Sequential()
    model.add(LSTM(50, activation = 'relu', batch_input_shape=(None, df.shape[1], df.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(256,activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation = 'sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    
    return model

def load_data(data_path_pattern):
    data_path = glob.glob(data_path_pattern)
    df = pd.read_csv(data_path[0],index_col = 'Date')

    return df

# 디렉토리, 디렉토리 내 파일 삭제
def delete_dir(dir_path, is_delete_top_dir=True):
    for root, dirs, files in os.walk(dir_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    if is_delete_top_dir:
        os.rmdir(dir_path)

RETURN_SUCCESS = 0
RETURN_FAILURE = -1
PREPROCESSED_DATA_PATTERN = "./preprocessed_data/*"
# Outoput Model Only
OUTPUT_MODEL_ONLY = False
# Output Model Directory
OUTPUT_MODEL_DIR = "./model"
# Output Model File Name
OUTPUT_MODEL_FILE = "model.h5"


def main():
    print("==================================================================")
    print("LSTM모델 학습")
    print("==================================================================")

    
    # 디렉토리 작성
    if not os.path.isdir(OUTPUT_MODEL_DIR):
        os.mkdir(OUTPUT_MODEL_DIR)
    delete_dir(OUTPUT_MODEL_DIR, False)

    
    #원본 데이터 로드
    df = load_data(PREPROCESSED_DATA_PATTERN)


    X_train, y_train, X_test, y_test = data_split(df)
    X_train_array = Scaler_to_np(X_train)
    X_test_array = Scaler_to_np(X_test)
    y_train_new = y_train[3::4]
    y_test_new = y_test[3::4]
    
    #시계열데이터 검증
    sequence_fold(X_train_array, y_train_new)

    #모델 학습
    model = gen_lstm_compile(X_train_array)
    model.fit(X_train_array, y_train_new, epochs = 10, batch_size = 64)
    pred = model.predict(X_test_array)
    pred = np.where(pred < 0.5, 0, 1)
    print(f"accuracy={accuracy_score(y_true=y_test_new,y_pred = pred)}")


    # 모델 저장
    model_file_path = os.path.join(OUTPUT_MODEL_DIR, OUTPUT_MODEL_FILE)
    model.save(model_file_path)


    return RETURN_SUCCESS

if __name__ == "__main__":
    main()