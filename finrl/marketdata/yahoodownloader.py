#coding=gbk
"""Contains methods and classes to collect data from
Yahoo Finance API
"""

import pandas as pd
import baostock as bs
import yfinance as yf
from .lxcUrl import hsDownloadData

class YahooDownloader:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API

    Attributes
    ----------
        start_date : str
            start date of the data (modified from config.py)
        end_date : str
            end date of the data (modified from config.py)
        ticker_list : list
            a list of stock tickers (modified from config.py)

    Methods
    -------
    fetch_data()
        Fetches data from yahoo API

    """

    def __init__(self, start_date: str, end_date: str, ticker_list: list):

        self.start_date = start_date
        self.end_date = end_date
        self.ticker_list = ticker_list

    def lxcDownload(self,tic):
        df = pd.read_csv("./"+"lxcData" + "/" + str(tic) + ".csv", index_col=0)
        date = df['date']
        df = df.drop("date",axis=1)
        print(df)
        df.index = pd.to_datetime(date)
        df.sort_index(inplace=True)
        df.index.name = "date"
        return df

    def fetch_data(self) -> pd.DataFrame:
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
        print('lxc:',len(self.ticker_list))
        lxc_temp = 1
        for tic in self.ticker_list:
            #print('正在下载第',lxc_temp,'个数据')
            lxc_temp = lxc_temp+1
            #temp_df = yf.download(tic, start=self.start_date, end=self.end_date)
            temp_df = self.lxcDownload(tic)
            print(temp_df)
            #temp_df = hsDownloadData(en_prod_code =tic, begin_date=self.start_date, end_date=self.end_date)
            #print('type temp_df is:', type(temp_df))
            #print('temp_df is:',temp_df)
            temp_df["tic"] = str(tic)
            data_df = data_df.append(temp_df)
        # reset the index, we want to use numbers as index instead of dates
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

        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)

        return data_df
    def lxc_fetch_data(self) -> pd.DataFrame:
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
        # 登陆系统
        def round_amount(vol):
            data = round(float(vol),2)
            return data
        lg = bs.login()
        # 显示登陆返回信息
        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)

        # 获取行业分类数据
        rs = bs.query_stock_industry()
        # rs = bs.query_stock_basic(code_name="浦发银行")
        print('query_stock_industry error_code:' + rs.error_code)
        print('query_stock_industry respond  error_msg:' + rs.error_msg)

        # 打印结果集
        lxc_list = []
        data_df = pd.DataFrame()
        while (rs.error_code == '0') & rs.next():
            # 获取一条记录，将记录合并在一起
            temp = rs.get_row_data()
            lxc_temp = temp
            if (temp[3] == "食品饮料"):
                lxc_list.append(temp[1])
                temp_df = bs.query_history_k_data_plus(temp[1], "date,open,high,low,close,volume", self.start_date, self.end_date).get_data()
                if(len(temp_df)<1):
                    continue
                temp_df["tic"] = str(temp[1])
                temp_df["open"] = temp_df["open"].apply(round_amount)
                temp_df["high"] = temp_df["high"].apply(round_amount)
                temp_df["low"] = temp_df["low"].apply(round_amount)
                temp_df["close"] = temp_df["close"].apply(round_amount)
                temp_df["volume"] = temp_df["volume"].apply(round_amount)
                data_df = data_df.append(temp_df)
        date = data_df["date"]
        data_df = data_df.drop("date",axis = 1)
        data_df.index = pd.to_datetime(date)
        data_df.index.name="date"
        print("data_df is:",data_df)
        data_df = data_df.reset_index()
        try:
            # convert the column names to standardized names
            data_df.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "tic",
            ]
            # use adjusted close price instead of close price
            #data_df["close"] = data_df["adjcp"]
            # drop the adjusted close price column
            #data_df = data_df.drop("adjcp", 1)
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
        data_df = data_df.sort_values(by=['date','tic']).reset_index(drop=True)
        return data_df

    def select_equal_rows_stock(self, df):
        df_check = df.tic.value_counts()
        df_check = pd.DataFrame(df_check).reset_index()
        df_check.columns = ["tic", "counts"]
        mean_df = df_check.counts.mean()
        equal_list = list(df.tic.value_counts() >= mean_df)
        names = df.tic.value_counts().index
        select_stocks_list = list(names[equal_list])
        df = df[df.tic.isin(select_stocks_list)]
        return df
