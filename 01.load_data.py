import os
import sys
import pandas_datareader.data as data

#주식 데이터 다운로드
def load_stock_price(keywords,dir_path):
    df = data.DataReader(keywords[2], 'yahoo',keywords[0],keywords[1])
    filename = os.path.join(dir_path,keywords[2])
    save_data(filename, df)

#다운로드 데이터 디렉토리에 저장
def save_data(filename, data):
    data.to_csv(filename + '.csv')

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
ORIGIN_DATA_DIR = "./origin_data"

def main():
    print("==================================================================")
    print("Stock Price Download")
    print("지정한 기간과 이름으로 검색된 주식 가격 다운로드")
    print("==================================================================")

    # 인수 체크
    argvs = sys.argv
    if len(argvs) != 2 or not argvs[1]:
        print("기간과 이름을 지정해 주세요. - 콤마(,)로 구분 가능")
        print("ex) 2004-07-01,2022-03-18,AAPL")
        return RETURN_FAILURE

    # 키워드 취득
    keywords = [x.strip() for x in argvs[1].split(',')]
    
    # 디렉토리 작성
    if not os.path.isdir(ORIGIN_DATA_DIR):
        os.mkdir(ORIGIN_DATA_DIR)
    delete_dir(ORIGIN_DATA_DIR, False)

    #주식데이터 다운로드
    load_stock_price(keywords, ORIGIN_DATA_DIR)

    return RETURN_SUCCESS

if __name__ == "__main__":
    main()