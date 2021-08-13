# -*- coding: utf-8 -*-
import requests
import datetime
import pandas as _pd
import base64
import time
import pandas as pd
import yfinance as yf

from tqdm import tqdm
import json
app_key = "81118a71-6e2d-4117-a03e-71c1e405faef"
app_secrect = "26b193b3-e8da-4eed-a613-147388f17acd"
token = 'B174810DA4D74108AFDE023141513F132021033114261881118A71'

def getEveryDay(begin_date,end_date):
    # 前闭后闭
    date_list = []
    begin_date = datetime.datetime.strptime(begin_date, "%Y-%m-%d")
    end_date = datetime.datetime.strptime(end_date,"%Y-%m-%d")
    while begin_date <= end_date:
        date_str = begin_date.strftime("%Y-%m-%d")
        date_list.append(date_str)
        begin_date += datetime.timedelta(days=1)
    return date_list
def getToken(app_key,app_secrect):
	global token
	bytesString = (app_key+':'+app_secrect).encode(encoding="utf-8")
	url = 'https://sandbox.hscloud.cn/oauth2/oauth2/token';
	header = {'Content-Type': 'application/x-www-form-urlencoded',
		'Authorization': 'Basic '+str(base64.b64encode(bytesString),encoding="utf-8")}
	field = {'grant_type' : 'client_credentials'}
	r = requests.post(url,data=field,headers=header)
	if r.json().get('access_token') :
		token = r.json().get('access_token')
		print("获取公共令牌:"+str(token))
		return
	else :
		print("获取公共令牌失败")
		exit
'''
def WriteFile(data):
    with open('AllStocksList.txt', 'a+') as f:
        for oneData in data:
            f.write(oneData+"\n")
        f.close()
'''
def fetch_data(start_date,end_date) -> pd.DataFrame:
    def addZero(str):
        result = str
        for i in range(len(str),4):
            result = "0"+result
        #深圳交易所的
        result ="00"+result+".sz"
        return result
    """Fetches data from Yahoo API
    Parameters
    ----------

    Returns
    -------
    `pd.DataFrame`
        7 columns: A date, open, high, low, close, volume and tick symbol
        for the specified stock ticker
    """
    # Download and save the data in a pandas DataFrame:
    data_df = pd.DataFrame()
    ticker_list = []
    for i in range(2000,3500):
        temp_str = addZero(str(i))
        ticker_list.append(temp_str)
    print('lxc:', len(ticker_list))
    #print(ticker_list)
    #lxc_temp = 1
    lxc_ticker_list = []
    with open('AllStocksList.txt', 'a+') as f:
        for tic in tqdm(ticker_list):
        #print('正在下载第', lxc_temp, '个数据')
        #lxc_temp = lxc_temp + 1
            temp_df = yf.download(tic, start=start_date, end=end_date)
        #print(temp_df)
            if(temp_df.empty):
                print("None!")
                continue
        # temp_df = hsDownloadData(en_prod_code =tic, begin_date=self.start_date, end_date=self.end_date)
        # print('type temp_df is:', type(temp_df))
        # print('temp_df is:',temp_df)
            f.write(tic+"\n")
            temp_df["tic"] = str(tic)
            data_df = data_df.append(temp_df)
    # reset the index, we want to use numbers as index instead of dates
    #print(lxc_ticker_list)
        f.close()
    data_df = data_df.reset_index()
    try:
        # convert the column names to standardized names
        data_df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
            "adjcp",
            "volume",
            "tic",
        ]
        # use adjusted close price instead of close price
        data_df["close"] = data_df["adjcp"]
        # drop the adjusted close price column
        data_df = data_df.drop("adjcp", 1)
    except NotImplementedError:
        print("the features are not supported currently")
    # create day of the week column (monday = 0)
    data_df["day"] = data_df["date"].dt.dayofweek
    # convert date to standard string format, easy to filter
    data_df["date"] = data_df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
    # drop missing data  Y
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    print("Shape of DataFrame: ", data_df.shape)
    # print("Display DataFrame: ", data_df.head())

    data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

    return data_df
def postOpenApi(url,params):
    global token
    header = {'Content-Type': 'application/x-www-form-urlencoded',
		'Authorization': 'Bearer '+token}
    r = requests.post(url,data=params,headers=header)
    temp = r.json().get('data')
    #print(temp[0]['high_price']=="")
    #print("result = "+str(r.json().get('data')))
    return temp
def hsDownloadData(en_prod_code,begin_date,end_date):
    dataList = getEveryDay(begin_date,end_date)
    url = "https://sandbox.hscloud.cn/gildataastock/v1/astock/quotes/daily_quote"
    Date =[]
    Open = []
    High = []
    Low = []
    Close = []
    Adj_Close = []
    Volume = []
    for oneDay in tqdm(dataList):
        #params = 'en_prod_code=600000.SH&trading_date=2016-12-30&unit=0'
        params = "en_prod_code="+en_prod_code+"&trading_date="+oneDay
        #print(params)
        temp = postOpenApi(url, params)
        if(temp[0]['high_price'] != ""):#有数据，开盘
            Date.append(temp[0]['trading_date'])
            Open.append(temp[0]['open_price'])
            High.append(temp[0]['high_price'])
            Low.append(temp[0]['low_price'])
            Close.append(temp[0]['close_price'])#后期需要修改
            Adj_Close.append(temp[0]['avg_price'])
            Volume.append(temp[0]['business_amount'])
        time.sleep(2)
    Frame = {"Open": Open,
             "High": High,
             "Low": Low,
             "Close": Close,
             "Adj Close": Adj_Close,
             "Volume": Volume

    }
    quotes =_pd.DataFrame.from_dict(Frame)
    quotes.index = _pd.to_datetime(Date)
    quotes.sort_index(inplace=True)
    quotes.index.name = "Date"
    return quotes
def jsonToDate(jsonDate):
    pass
if __name__ == '__main__':
    #getToken(app_key,app_secrect)
    '''
    data = hsDownloadData('600918','2020-12-21','2021-01-01')
    print(len(data))
    data = hsDownloadData('000001.SZ', '2020-12-21', '2021-01-01')
    print(len(data))
    print(type(data))
    '''
    #fetch_data("2021-03-21",'2021-04-14')
    a=[1,2,3,4,5,6,7,8,9,0]
    for i in range(0,len(a),4):
        print(a[i])