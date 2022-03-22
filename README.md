# stockprice_predict
주식 가격 예측\
(월~목 데이터로 금요일 가격이 상승할 것인지 말 것인지 예측하는 모델)\

01.load_data.py  : pandas_datareader를 통해 야후에서 주식 데이터 다운로드 후 저장\
02.preprocess.py : 다운로드 받은 데이터를 파생변수 생성 및 전처리 후 저장\
03.modeling.py : LSTM 모델 생성\
04.predict.py : 2월,3월 금요일 주식가격 예측\


개선사항
- 모델의 성능이 test_data(2020-2021년)의 경우 최대 56%정도 밖에 나오지 않음\
- Django 및 css파일로 웹페이지 생성해보기\
