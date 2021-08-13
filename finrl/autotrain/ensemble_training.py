import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import preprocessing

matplotlib.use("Agg")
import datetime
import torch

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.env.lxc_env_stocktrading import lxcStockTradingEnv
from finrl.model.models import DRLAgent
from finrl.model.models import DRLEnsembleAgent
from finrl.trade.backtest import backtest_stats as BackTestStats
from stable_baselines3 import A2C

def train_one():
    """
    train an agent
    """
    print('here')
    print("==============Start Fetching Data===========")
    '''
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    '''
    #names=["date","open","high","low","close","volume","tic","day",]
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210315-07h382" + ".csv",index_col=0)
    print('GPU is :',torch.cuda.is_available())
    all_model = []
    df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210330-08h52" + ".csv", index_col=0)
    #print(df)
    print("==============Start Feature Engineering===========")

    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    print('fe is:',fe)
    processed = fe.preprocess_data(df)
    print('processed is:', processed)
    # Training & Trading data split
    #train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    #trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    #print('trade is:',trade)
    # calculate state action space
    stock_dimension = 30
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )

    env_kwargs = {
        "hmax": 100,
        "initial_amount": 1000000,
        "buy_cost_pct": 0.001,
        "sell_cost_pct": 0.001,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST,
        "action_space": stock_dimension,
        "reward_scaling": 1e-4
        }

    train_period = [config.START_DATE,config.START_TRADE_DATE]
    val_test_period = [config.START_TRADE_DATE,config.END_DATE]
    ensembleAgent = DRLEnsembleAgent(df = processed,train_period = train_period,val_test_period=val_test_period,rebalance_window=10,
    validation_window=10,print_verbosity=True,**env_kwargs)
    timesteps_dict={'a2c': 10000,'ppo':10000,'ddpg':10000}
    lxcModel = ensembleAgent.run_ensemble_strategy(A2C_model_kwargs=config.A2C_PARAMS,PPO_model_kwargs=config.PPO_PARAMS,DDPG_model_kwargs=config.DDPG_PARAMS,timesteps_dict=timesteps_dict)



