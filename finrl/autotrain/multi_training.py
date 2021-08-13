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
from finrl.env.multi_env_stocktrading import StockTradingEnv

from finrl.lxcalgorithms.gateway import Gateway
from finrl.model.multi_models import DRLAgent
from finrl.trade.backtest import backtest_stats as BackTestStats

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

def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    print('GPU is :', torch.cuda.is_available())
    start_date = config.START_DATE,
    end_date = config.START_TRADE_DATE,
    start_date = start_date[0]
    end_date = end_date[0]
    date_list = getEveryDay(start_date,end_date)
    food = Gateway()
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    for i in range(0,1):
        df = YahooDownloader(
        start_date=start_date,
        end_date=end_date,
        ticker_list=config.DOW_30_TICKER,
        ).lxc_fetch_data()
        print("==============Start Feature Engineering===========")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=True,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(df)
        train = data_split(processed, start_date, end_date)
        stock_dimension = len(train.tic.unique())
        #print("train.tic.unique() is:")
        #print(train.tic.unique())
        print('stock_dimension is:', stock_dimension)
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
            # "action_space": stock_dimension,
            "action_space": 1,
            "reward_scaling": 1e-4
        }
        e_train_gym = StockTradingEnv(df=train, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        agent = DRLAgent(env=env_train)
        print("==============Model Training===========")

        print("start training ddpg model")
        ##################################################################################################a2c
        print("start pre_training model")
        multi_number = stock_dimension
        #temp = agent.get_model(model_name="a2c", lxc_stock_number=0)
        #agent.train_model(model=temp, tb_log_name="a2c", total_timesteps=1000)

        #print("e_train_gym.normal_high is:", e_train_gym.normal_high)
        #print("e_train_gym.normal_low is:", e_train_gym.normal_low)
        #e_trade_gym.normal_high = e_train_gym.normal_high
        #e_trade_gym.normal_low = e_train_gym.normal_low
        print("start main_training model")
        for j in range(0, multi_number):
            if(j+1>food.agents_number):

                model_a2c = agent.get_model(model_name="a2c", lxc_stock_number=j,all_stock_number = multi_number)
                model_a2c.all_stock_number = multi_number
                model_a2c.env.reset()
                if(j!=0):
                    trained_a2c = agent.get_pre_model(model=model_a2c, tb_log_name="a2c", total_timesteps=1000,
                                                      lxcName="lxcMulti" + str(j - 1))
                    trained_a2c.lxc_stock_number = j
                    trained_a2c.all_stock_number = multi_number
                    trained_a2c = trained_a2c.online_learning(total_timesteps=1000, tb_log_name="a2c")
                    agent.save_pre_model(model=trained_a2c, tb_log_name="a2c", total_timesteps=1000,
                                         lxcName="lxcMulti" + str(j))
                else:
                    trained_a2c = agent.train_lxc_model(
                    # model=model_a2c, tb_log_name="a2c", total_timesteps=80000,lxcType=1,lxcName="lxc2"
                        model=model_a2c, tb_log_name="a2c", total_timesteps=1000, lxcType= None, lxcName="lxcMulti" + str(j)
                        )
                food.agents.append(trained_a2c)
                print(j,"'s model is trained done")
            else:
                print("here!!!")
                food.agents[j].all_stock_number = multi_number
                food.agents[j].env = env_train
                food.agents[j] = food.agents[j].online_learning(total_timesteps=1000, tb_log_name="a2c")
                env_train.reset()
        food.agents_number = multi_number



    # names=["date","open","high","low","close","volume","tic","day",]
    # df = pd.read_csv("./" + config.DATA_SAVE_DIR + "/" + "20210315-07h382" + ".csv",index_col=0)
    print('GPU is :', torch.cuda.is_available())

    #########################################################################
    print("==============Start Trading===========")
    start_date = config.START_TRADE_DATE,
    end_date = config.END_DATE,
    df_account_value  = pd.DataFrame()
    df_actions  = pd.DataFrame()
    start_date = start_date[0]
    end_date = end_date[0]
    date_list = getEveryDay(start_date, end_date)
    time_step = 30
    #for i in range(0, len(date_list) - time_step, time_step):
    for i in range(0, 1):
        df = YahooDownloader(
            start_date=start_date,
            end_date=end_date,
            ticker_list=config.DOW_30_TICKER,
        ).lxc_fetch_data()
        print("==============Start Feature Engineering===========")
        fe = FeatureEngineer(
            use_technical_indicator=True,
            tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
            use_turbulence=True,
            user_defined_feature=False,
        )
        processed = fe.preprocess_data(df)
        train = data_split(processed, start_date, end_date)
        stock_dimension = len(train.tic.unique())
        # print("train.tic.unique() is:")
        # print(train.tic.unique())
        print('stock_dimension is:', stock_dimension)
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
            # "action_space": stock_dimension,
            "action_space": 1,
            "reward_scaling": 1e-4
        }
        e_train_gym = StockTradingEnv(df=train,turbulence_threshold=250.0, **env_kwargs)
        env_train, _ = e_train_gym.get_sb_env()
        agent = DRLAgent(env=env_train)
        print("==============Model Training===========")

        print("start training ddpg model")

        ##################################################################################################a2c
        print("start pre_training model")
        multi_number = stock_dimension
        # temp = agent.get_model(model_name="a2c", lxc_stock_number=0)
        # agent.train_model(model=temp, tb_log_name="a2c", total_timesteps=1000)

        # print("e_train_gym.normal_high is:", e_train_gym.normal_high)
        # print("e_train_gym.normal_low is:", e_train_gym.normal_low)
        # e_trade_gym.normal_high = e_train_gym.normal_high
        # e_trade_gym.normal_low = e_train_gym.normal_low
        print("start main_training model")
        #################################检测是否有新股票上市##################################
        for j in range(food.agents_number,multi_number):
            model_a2c = agent.get_model(model_name="a2c", lxc_stock_number=j, all_stock_number=multi_number)
            model_a2c.all_stock_number = multi_number
            model_a2c.env.reset()
            trained_a2c = agent.get_pre_model(model=model_a2c, tb_log_name="a2c", total_timesteps=1000,lxcName="lxcMulti" + str(food.agents_number-1))
            trained_a2c.lxc_stock_number = j
            trained_a2c.all_stock_number = multi_number
            food.agents.append(trained_a2c)
        food.agents_number = multi_number
        #################################测试##################################
        temp_account_value, temp_actions = DRLAgent.Multi_DRL_prediction(
            model=food.agents, environment=e_train_gym,all_stock_number=multi_number
        )
        #print("temp_account_value is:",temp_account_value)
        #print("temp_actions is:", temp_actions)
        df_account_value = df_account_value.append(temp_account_value)
        df_actions = df_actions.append(temp_actions)
        #print("df_account_value is:", df_account_value)
        #print("df_actions is:", df_actions)
        #exit(0)

        ################################再训练######################################
        '''
        for j in range(0, multi_number):
            food.agents[j].all_stock_number = multi_number
            env_train.reset()
            food.agents[j].env = env_train
            food.agents[j] = food.agents[j].online_learning(total_timesteps=1000, tb_log_name="a2c")
            agent.save_pre_model(model=model_a2c, tb_log_name="a2c", total_timesteps=1000,lxcName="lxcMulti" + str(j))
            #env_train.reset()
        '''

    '''
    df_account_value, df_actions = DRLAgent.DRL_prediction(
        model=multi_A2C, environment=e_trade_gym,lxc_env = e_train_gym
        # model=trained_ddpg, environment=e_trade_gym
    )
    '''
    print("df_account_value is:",df_account_value)
    print("df_actions is:",df_actions)
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/df_account_value_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/df_actions_" + now + ".csv")

    print("==============Get Backtest all Results===========")
    perf_stats_all = BackTestStats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/perf_stats_all_" + now + ".csv")

