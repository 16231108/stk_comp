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
from finrl.trade.backtest import backtest_stats as BackTestStats
from stable_baselines3 import A2C

def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=config.DOW_30_TICKER,
    ).fetch_data()
    
    #names=["date","open","high","low","close","volume","tic","day",]
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210315-07h382" + ".csv",index_col=0)
    print('GPU is :',torch.cuda.is_available())
    all_model = []
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210330-08h52" + ".csv", index_col=0)
    #df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210331-21h50" + ".csv", index_col=0)
    #print(df)
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )
    print(df)

    processed = fe.preprocess_data(df)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)
    print('trade is:',trade)
    print('train is:', train)
    # calculate state action space
    stock_dimension = len(train.tic.unique())
    print('stock_dimension is:',stock_dimension)
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

    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    #lxc_trade_gym =lxcStockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs,lxcModels=all_model)
    e_trade_gym2 = StockTradingEnv(df=trade, turbulence_threshold=250.0, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    #lxc_env_train, _ = lxc_trade_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()
    agent = DRLAgent(env=env_train)
    #lxc_agent = DRLAgent(env=lxc_env_train)

    print("==============Model Training===========")

    print("start training ddpg model")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")


    ###################################################################################################a2c
    print("start training a2c model")
    model_sac = agent.get_model("sac")
    trained_sac = agent.train_lxc_model(
        #model=model_a2c, tb_log_name="a2c", total_timesteps=80000,lxcType=1,lxcName="lxc2"
        model=model_sac, tb_log_name="sac", total_timesteps=1800, lxcType=None, lxcName="lxc3"
    )
    #print('trained_a2c is:', trained_a2c)
    all_model.append(trained_sac)

    
    ####################################################################sac

    '''
    print("start training sac model")
    model_sac = agent.get_model("sac")
    trained_sac = agent.train_lxc_model(
        #model=model_sac, tb_log_name="sac", total_timesteps=180000, lxcType=1, lxcName="lxc1"
        model = model_sac, tb_log_name = "sac", total_timesteps = 80000, lxcType = 1, lxcName = "lxc2"
    )    
    #print('trained_sac is:', trained_sac)
    all_model.append(trained_sac)
    '''
    '''
    #################################################################ddpg

    model_ddpg = agent.get_model("ddpg")
    trained_ddpg = agent.train_lxc_model(
        model=model_ddpg, tb_log_name="ddpg", total_timesteps=180000,lxcType=1,lxcName="lxc1"
    )
    #print('trained_ddpg is:',trained_ddpg)
    all_model.append(trained_ddpg)
    '''
    ###################################################################lxca2c

    '''
    print("start training lxcA2C model")
    model_lxcSAC = lxc_agent.get_model("lxcDDPG")
    trained_lxcSAC = lxc_agent.train_lxc_model(
        model=model_lxcSAC, tb_log_name="lxcDDPG", total_timesteps=4000, lxcType=1, lxcName="lxc2"
        #model=model_lxcSAC, tb_log_name="lxcSAC", total_timesteps=80000, lxcType=1, lxcName="lxc2"
        #model=model_lxcSAC, tb_log_name="lxcA2C", total_timesteps=80000, lxcType=1, lxcName="lxc2"
        #model=model_lxcSAC, tb_log_name="lxcDDPG", total_timesteps=80000, lxcType=None, lxcName="lxc2"
    )
    all_model.append(trained_lxcSAC)
    '''

    #########################################################################
    print("==============Start Trading===========")
    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, test_data=trade, test_env=env_trade, test_obs=obs_trade
    )
    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=trained_sac, environment=e_trade_gym
        #model=trained_ddpg, environment=e_trade_gym
    )

    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")



    print("==============Get Backtest all Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

