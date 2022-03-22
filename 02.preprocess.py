import os
import pathlib
import glob

import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

def preprocessing(dir_path, df):
    #날짜별로 중복되는 데이터 제거
    df = df.drop_duplicates(subset='Date', keep="last")

    df['Date2'] = df['Date'] #나중에 필요하기때문에 복사

    #이동평균 칼럼 생성
    close = df['Adj Close']
    df['5MA'] = close.rolling(window = 5).mean()
    df['10MA'] = close.rolling(window = 10).mean()
    df['20MA'] = close.rolling(window = 20).mean()
    df['60MA'] = close.rolling(window = 60).mean()
    df['120MA'] = close.rolling(window = 120).mean()

    #골든크로스와 데드크로스 값 파악을 위한 5일 이동평균 값과 20일 이동평균 값의 가격 차이
    df['cross'] = df['5MA'] - df['20MA']

    df['Date'] = pd.to_datetime(df['Date']) #Date 칼럼의 type 변경
    df['Date2'] = pd.to_datetime(df['Date2'])
    df['weekday'] = df['Date'].dt.weekday #요일 칼럼 생성
    df = df.sort_values(by = 'Date') # 가장 오래된 날짜를 먼저 표시되도록 정렬
    df.set_index('Date', inplace = True) #Date를 index로

    # 다음날의 종가하고 당일의 종가하고의 차를 계산하여 새로운 칼럼으로 추가
    df_shift = df.shift(-1)
    df['Gap_close'] = df_shift['Close'] - df['Close']

    #새로운 컬럼 Up을 추가하고, 다음날 시작가가 올라갈 것 같으면 1, 내려갈 것 같으면 0
    df['Up'] = df['Gap_close'].apply(lambda x: 1 if x >= 0 else 0)
    df.drop(['Gap_close'], axis = 1,inplace = True)

    #전날과 비교했을 때 종가 비율
    df_shift = df.shift(1)
    df['Close_ratio'] = (df['Close'] - df_shift['Close']) / df_shift['Close']

    df['Body'] = df['Open'] - df['Close'] #시작 -종가

    df.dropna(inplace = True) #NUll값 제거

    #데이터가 월요일부터 시작하도록 변경
    start_index = df[df['weekday'] == 0].index[0]
    df = df.loc[start_index:]

    start = df[df['weekday'] == 0].index[0]   #주차 데이터 생성
    df['weeks'] = (df['Date2'] - start) // timedelta(weeks = 1)
    
    # #월~목 데이터로 금요일을 예측하기 위해 휴일이 있는 데이터 삭제
    # list_weeks = []
    # list_weeks = df['weeks'].unique()
    
    # df['week_days'] = 0 # 각 주별로 일수가 몇개인지 확인
    # for i in list_weeks:
    #     df['week_days'][df['weeks'] == i] = len(df[df['weeks'] == i])
    # df = df[df['week_days'] ==5]
    #df.drop(['weeks','week_days','Date2'], axis = 1, inplace = True) #필요없는 칼럼 제거
    df.drop(['weeks','Date2'], axis = 1, inplace = True) #필요없는 칼럼 제거

    filename = os.path.join(dir_path, 'preprocessed_data')
    save_data(filename, df)


def save_data(filename, data):
    data.to_csv(filename + '.csv')

def load_data(data_path_pattern):
    data_path = glob.glob(data_path_pattern)
    df = pd.read_csv(data_path[0])

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
ORIGIN_DATA_PATTERN = "./origin_data/*"
PREPROCESSED_DATA_DIR = "./preprocessed_data"


def main():
    print("==================================================================")
    print("주식 데이터 전처리")
    print("==================================================================")

    
    # 디렉토리 작성
    if not os.path.isdir(PREPROCESSED_DATA_DIR):
        os.mkdir(PREPROCESSED_DATA_DIR)
    delete_dir(PREPROCESSED_DATA_DIR, False)
    
    #원본 데이터 로드
    df = load_data(ORIGIN_DATA_PATTERN)

    #데이터 전처리
    preprocessing(PREPROCESSED_DATA_DIR, df)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()