# common library
import pandas as pd
import numpy as np
import time
import gym
import os

# RL models from stable-baselines
# from stable_baselines import SAC
# from stable_baselines import TD3
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
import torch as th

import tensorflow as tf

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from finrl.config import config
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv

from finrl.lxcalgorithms.multiA2C import A2C
from finrl.lxcalgorithms.lxcA2C import lxcA2C
from finrl.lxcalgorithms.lxcSAC import lxcSAC
from finrl.lxcalgorithms.lxcDDPG import lxcDDPG
from stable_baselines3 import PPO
from stable_baselines3 import TD3
import random
from stable_baselines3.td3.policies import MlpPolicy
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
'''
th.enable_eager_execution(
    config=None,
    device_policy=None,
    execution_mode=None
)
'''

from stable_baselines3 import SAC


MODELS = {"a2c": A2C, "ddpg": DDPG, "td3": TD3, "sac": SAC, "ppo": PPO,"lxcA2C": lxcA2C,"lxcSAC": lxcSAC,"lxcDDPG": lxcDDPG,}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}


class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
        train_PPO()
            the implementation for PPO algorithm
        train_A2C()
            the implementation for A2C algorithm
        train_DDPG()
            the implementation for DDPG algorithm
        train_TD3()
            the implementation for TD3 algorithm
        train_SAC()
            the implementation for SAC algorithm
        DRL_prediction()
            make a prediction in a test dataset and get results
    """
    '''   
    def DRL_prediction(model, environment):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            action, _states = model.predict(test_obs)
            #account_memory = test_env.env_method(method_name="save_asset_memory")
            #actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
              account_memory = test_env.env_method(method_name="save_asset_memory")
              actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]
    '''

    def readFile(filename):
        with open(str(filename), 'r') as f:
            data = f.readlines()
        return data

    def writeFile(filename, data):
        with open(filename, 'w') as f:
            for oneData in data:
                f.write(str(oneData) + '\n')
        print('File had been writed Succed!')

    def Multi_DRL_prediction(model, environment, all_stock_number):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        for i in range(len(environment.df.index.unique())):
            lxc_action = []
            new_obs = test_obs
            lxc_state = new_obs[0][0]
            lxc_data_close = new_obs[0][1:all_stock_number+1]
            lxc_state_list = new_obs[0][all_stock_number+1:2*all_stock_number+1]
            lxc_temp = new_obs[0][2*all_stock_number+1:]
            for j in range(0, len(model)):
                lxc_new_obs = []
                lxc_new_obs.append(lxc_state)
                lxc_new_obs.append(lxc_data_close[j])
                if(lxc_data_close[j] < 0.01):
                    lxc_action.append(0)
                    print("new stock in this")
                    continue
                lxc_new_obs.append(lxc_state_list[j])
                for lxc_i in range(0, len(lxc_temp)):
                    if (lxc_i) % (all_stock_number) == j:
                        lxc_new_obs.append(lxc_temp[lxc_i])
                #print("old test_obs length is:", len(test_obs[0]))
                #print("lxc_state_list:", lxc_state_list)
                #exit(0)
                #print("all_stock_number is:", all_stock_number)
                test_obs = np.array([lxc_new_obs])

                # action,_ = oneModel.predict(test_obs)
                #print("new test_obs is:", test_obs)
                action, _ = model[j].predict(test_obs)
                lxc_action.append(action[0][0])
            action = np.array([lxc_action])
            #all_action += lxc_action
            #print("action is:", action)
            #print("value is:", lxc_value)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, reward, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    @staticmethod
    def DRL_prediction(model, environment,lxc_env):
        test_env, test_obs = environment.get_sb_env()
        """make a prediction"""
        account_memory = []
        actions_memory = []
        test_env.reset()
        print('environment.df is:',environment.df)
        lxc_dataFrame =lxc_env.df
        #设置n_滑窗更新agent
        n_month = 1
        print("lxc_dataFrame.columns.size is:",len(lxc_dataFrame))
        lxc_dataFrame = lxc_dataFrame[len(lxc_dataFrame)-n_month*6*30:]
        print("lxc_dataFrame is:", lxc_dataFrame)
        lxc_dataFrame.index = lxc_dataFrame.index - 380
        #lxc_dataFrame.index = lxc_dataFrame.index - 384
        for i in range(len(environment.df.index.unique())):
            temp_dataFrame = environment.df.loc[i, :]
            lxc_dataFrame = lxc_dataFrame[6:]
            lxc_dataFrame.index = lxc_dataFrame.index - 1
            temp_dataFrame.index = temp_dataFrame.index + n_month*30-1-i
            lxc_dataFrame = pd.concat([lxc_dataFrame, temp_dataFrame], ignore_index=False)
            if(i%500==0 and i!=0):
                print("lxc_dataFrame is:",lxc_dataFrame)
                stock_dimension = 6
                #print('stock_dimension is:', stock_dimension)
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
                lxc_env = StockTradingEnv(df=lxc_dataFrame, **env_kwargs)
                for j in range(0,len(model)):
                    #lxc_env.reset()
                    env_train, _ = lxc_env.get_sb_env()
                    model[j].env = env_train
                    model[j] = model[j].online_learning(total_timesteps=10000,tb_log_name="a2c")
                    env_train.reset()

            lxc_action = []
            lxc_value=[]
            new_obs = test_obs
            lxc_state = new_obs[0][0]
            lxc_data_close = new_obs[0][1:7]
            lxc_state_list = new_obs[0][7:13]
            lxc_temp = new_obs[0][13:]
            for j in range(0,len(model)):
                lxc_new_obs = []
                lxc_new_obs.append(lxc_state)
                lxc_new_obs.append(lxc_data_close[j])
                lxc_new_obs.append(lxc_state_list[j])
                for lxc_i in range(0, len(lxc_temp)):
                    if (lxc_i) % (6) == j:
                        lxc_new_obs.append(lxc_temp[lxc_i])
                #print("old test_obs is:", test_obs)
                test_obs = np.array([lxc_new_obs])

                #action,_ = oneModel.predict(test_obs)
                print("new test_obs is:",test_obs)

                action, _ = model[j].predict(test_obs)
                lxc_action.append(action[0][0])


                '''
                print("temp_obs is:", test_obs)
                temp_obs = th.as_tensor(test_obs).to(th.device("cpu"))
                temp_action = th.as_tensor(action).to(th.device("cpu"))
                #print("temp is:",temp_obs)
                
                temp_value, log_prob, entropy = model[j].policy.evaluate_actions(temp_obs, temp_action)
                temp_value = temp_value.flatten()
                print("temp_value is:", temp_value)
                lxc_value.append(temp_value)
                if(temp_value<0):
                    lxc_action.append(0)
                else:
                    lxc_action.append(0)
                '''
            action = np.array([lxc_action])
            print("action is:",action)
            print("value is:", lxc_value)
            # account_memory = test_env.env_method(method_name="save_asset_memory")
            # actions_memory = test_env.env_method(method_name="save_action_memory")
            test_obs, rewards, dones, info = test_env.step(action)
            if i == (len(environment.df.index.unique()) - 2):
                account_memory = test_env.env_method(method_name="save_asset_memory")
                actions_memory = test_env.env_method(method_name="save_action_memory")
            if dones[0]:
                print("hit end!")
                break
        return account_memory[0], actions_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        lxc_stock_number,
        all_stock_number,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            lxc_stock_number=lxc_stock_number,
            all_stock_number = all_stock_number,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **model_kwargs,
        )
        return model

    def train_model(self, model, tb_log_name, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        #print(model.__class__)
        #model.save(f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}lxc2")
        #lxcModel = model.load(f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}")
        return model
    def get_pre_model(self, model, tb_log_name, total_timesteps=5000,lxcName = None):
        model = model.load(
            f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}{lxcName}",env = model.env)
        #print("one model train done!")
        return model
    def save_pre_model(self,model, tb_log_name, total_timesteps=5000,lxcName = None):
        model.save(
            f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}{lxcName}")

    def train_lxc_model(self, model, tb_log_name, total_timesteps=5000,lxcType=None,lxcName = None):
        if lxcType is None:
            model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
            model.save(                f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}{lxcName}")
            print("one model train done!")
            return model
        #print(model.__class__)
        #model.save(f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}lxc2")
        #print("lxcModel env is:", model.env)
        lxcModel = model.load(f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}{lxcName}",env = model.env)
        #print("lxcModel env is:",lxcModel.env)
        lxcModel.num_timesteps = model.num_timesteps
        #lxcModel = lxcModel.online_learning(total_timesteps=total_timesteps,tb_log_name=tb_log_name)
        #lxcModel.save(
            #f"{config.TRAINED_MODEL_DIR}/{tb_log_name.upper()}_{total_timesteps // 1000}k_{total_timesteps}{lxcName}")
        #print("one model train done!")
        #lxcModel.env.reset()
        return lxcModel


class DRLEnsembleAgent:
    @staticmethod
    def get_model(model_name,
                    env,
                    policy="MlpPolicy",
                    policy_kwargs=None,
                    model_kwargs=None,
                    verbose=1):

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[temp_model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(temp_model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            **temp_model_kwargs,
        )
        return model

    @staticmethod
    def train_model(model, model_name, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name)
        model.save(f"{config.TRAINED_MODEL_DIR}/{model_name.upper()}_{total_timesteps//1000}k_{iter_num}")
        return model

    @staticmethod
    def get_validation_sharpe(iteration,model_name):
        ###Calculate Sharpe ratio based on validation results###
        df_total_value = pd.read_csv('results/account_value_validation_{}_{}.csv'.format(model_name,iteration))
        sharpe = (4 ** 0.5) * df_total_value['daily_return'].mean() / \
                 df_total_value['daily_return'].std()
        return sharpe

    def __init__(self,df,
                train_period,val_test_period,
                rebalance_window, validation_window,
                stock_dim,
                hmax,
                initial_amount,
                buy_cost_pct,
                sell_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                print_verbosity):

        self.df=df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[(df.date > val_test_period[0])&(df.date <= val_test_period[1])].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity


    def DRL_validation(self, model, test_data, test_env, test_obs):
        ###validation process###
        for i in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(self,model,name,last_state,iter_num,turbulence_threshold,initial):
        ### make a prediction based on trained model###

        ## trading env
        trade_data = data_split(self.df, start=self.unique_trade_date[iter_num - self.rebalance_window], end=self.unique_trade_date[iter_num])
        trade_env = DummyVecEnv([lambda: StockTradingEnv(trade_data,
                                                        self.stock_dim,
                                                        self.hmax,
                                                        self.initial_amount,
                                                        self.buy_cost_pct,
                                                        self.sell_cost_pct,
                                                        self.reward_scaling,
                                                        self.state_space,
                                                        self.action_space,
                                                        self.tech_indicator_list,
                                                        turbulence_threshold=turbulence_threshold,
                                                        initial=initial,
                                                        previous_state=last_state,
                                                        model_name=name,
                                                        mode='trade',
                                                        iteration=iter_num,
                                                        print_verbosity=self.print_verbosity)])

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.render()

        df_last_state = pd.DataFrame({'last_state': last_state})
        df_last_state.to_csv('results/last_state_{}_{}.csv'.format(name, i), index=False)
        return last_state

    def run_ensemble_strategy(self,A2C_model_kwargs,PPO_model_kwargs,DDPG_model_kwargs,timesteps_dict):
        """Ensemble Strategy that combines PPO, A2C and DDPG"""
        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        ppo_sharpe_list = []
        ddpg_sharpe_list = []
        a2c_sharpe_list = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[(self.df.date<self.train_period[1]) & (self.df.date>=self.train_period[0])]
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, .90)

        start = time.time()
        for i in range(self.rebalance_window + self.validation_window, len(self.unique_trade_date), self.rebalance_window):
            validation_start_date = self.unique_trade_date[i - self.rebalance_window - self.validation_window]
            validation_end_date = self.unique_trade_date[i - self.rebalance_window]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            ## initial state is empty
            if i - self.rebalance_window - self.validation_window == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False

            # Tuning trubulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
            end_date_index = self.df.index[self.df["date"] == self.unique_trade_date[i - self.rebalance_window - self.validation_window]].to_list()[-1]
            start_date_index = end_date_index - 63 + 1

            historical_turbulence = self.df.iloc[start_date_index:(end_date_index + 1), :]

            historical_turbulence = historical_turbulence.drop_duplicates(subset=['date'])

            historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)

            print(historical_turbulence_mean)

            if historical_turbulence_mean > insample_turbulence_threshold:
                # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
                # then we assume that the current market is volatile,
                # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
                # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
                turbulence_threshold = insample_turbulence_threshold
            else:
                # if the mean of the historical data is less than the 90% quantile of insample turbulence data
                # then we tune up the turbulence_threshold, meaning we lower the risk
                turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)
            print("turbulence_threshold: ", turbulence_threshold)

            ############## Environment Setup starts ##############
            ## training env
            train = data_split(self.df, start=self.train_period[0], end=self.unique_trade_date[i - self.rebalance_window - self.validation_window])
            self.train_env = DummyVecEnv([lambda: StockTradingEnv(train,
                                                                self.stock_dim,
                                                                self.hmax,
                                                                self.initial_amount,
                                                                self.buy_cost_pct,
                                                                self.sell_cost_pct,
                                                                self.reward_scaling,
                                                                self.state_space,
                                                                self.action_space,
                                                                self.tech_indicator_list,
                                                                print_verbosity=self.print_verbosity)])

            validation = data_split(self.df, start=self.unique_trade_date[i - self.rebalance_window - self.validation_window],
                                    end=self.unique_trade_date[i - self.rebalance_window])
            ############## Environment Setup ends ##############

            ############## Training and Validation starts ##############
            print("======Model training from: ", self.train_period[0], "to ",
                  self.unique_trade_date[i - self.rebalance_window - self.validation_window])
            # print("training: ",len(data_split(df, start=20090000, end=test.datadate.unique()[i-rebalance_window]) ))
            # print("==============Model Training===========")
            print("======A2C Training========")
            model_a2c = self.get_model("a2c",self.train_env,policy="MlpPolicy",model_kwargs=A2C_model_kwargs)
            model_a2c = self.train_model(model_a2c, "a2c", tb_log_name="a2c_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['a2c']) #100_000

            print("======A2C Validation from: ", validation_start_date, "to ",validation_end_date)
            val_env_a2c = DummyVecEnv([lambda: StockTradingEnv(validation,
                                                                self.stock_dim,
                                                                self.hmax,
                                                                self.initial_amount,
                                                                self.buy_cost_pct,
                                                                self.sell_cost_pct,
                                                                self.reward_scaling,
                                                                self.state_space,
                                                                self.action_space,
                                                                self.tech_indicator_list,
                                                                turbulence_threshold=turbulence_threshold,
                                                                iteration=i,
                                                                model_name='A2C',
                                                                mode='validation',
                                                                print_verbosity=self.print_verbosity)])
            val_obs_a2c = val_env_a2c.reset()
            self.DRL_validation(model=model_a2c,test_data=validation,test_env=val_env_a2c,test_obs=val_obs_a2c)
            sharpe_a2c = self.get_validation_sharpe(i,model_name="A2C")
            print("A2C Sharpe Ratio: ", sharpe_a2c)

            print("======PPO Training========")
            model_ppo = self.get_model("ppo",self.train_env,policy="MlpPolicy",model_kwargs=PPO_model_kwargs)
            model_ppo = self.train_model(model_ppo, "ppo", tb_log_name="ppo{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ppo']) #100_000
            print("======PPO Validation from: ", validation_start_date, "to ",validation_end_date)
            val_env_ppo = DummyVecEnv([lambda: StockTradingEnv(validation,
                                                                self.stock_dim,
                                                                self.hmax,
                                                                self.initial_amount,
                                                                self.buy_cost_pct,
                                                                self.sell_cost_pct,
                                                                self.reward_scaling,
                                                                self.state_space,
                                                                self.action_space,
                                                                self.tech_indicator_list,
                                                                turbulence_threshold=turbulence_threshold,
                                                                iteration=i,
                                                                model_name='PPO',
                                                                mode='validation',
                                                                print_verbosity=self.print_verbosity)])
            val_obs_ppo = val_env_ppo.reset()
            self.DRL_validation(model=model_ppo,test_data=validation,test_env=val_env_ppo,test_obs=val_obs_ppo)
            sharpe_ppo = self.get_validation_sharpe(i,model_name="PPO")
            print("PPO Sharpe Ratio: ", sharpe_ppo)

            print("======DDPG Training========")
            model_ddpg = self.get_model("ddpg",self.train_env,policy="MlpPolicy",model_kwargs=DDPG_model_kwargs)
            model_ddpg = self.train_model(model_ddpg, "ddpg", tb_log_name="ddpg_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ddpg'])  #50_000
            print("======DDPG Validation from: ", validation_start_date, "to ",validation_end_date)
            val_env_ddpg = DummyVecEnv([lambda: StockTradingEnv(validation,
                                                                self.stock_dim,
                                                                self.hmax,
                                                                self.initial_amount,
                                                                self.buy_cost_pct,
                                                                self.sell_cost_pct,
                                                                self.reward_scaling,
                                                                self.state_space,
                                                                self.action_space,
                                                                self.tech_indicator_list,
                                                                turbulence_threshold=turbulence_threshold,
                                                                iteration=i,
                                                                model_name='DDPG',
                                                                mode='validation',
                                                                print_verbosity=self.print_verbosity)])
            val_obs_ddpg = val_env_ddpg.reset()
            self.DRL_validation(model=model_ddpg,test_data=validation,test_env=val_env_ddpg,test_obs=val_obs_ddpg)
            sharpe_ddpg = self.get_validation_sharpe(i,model_name="DDPG")

            ppo_sharpe_list.append(sharpe_ppo)
            a2c_sharpe_list.append(sharpe_a2c)
            ddpg_sharpe_list.append(sharpe_ddpg)

            print("======Best Model Retraining from: ", self.train_period[0], "to ",
                  self.unique_trade_date[i - self.rebalance_window])
            # Environment setup for model retraining up to first trade date
            train_full = data_split(self.df, start=self.train_period[0], end=self.unique_trade_date[i - self.rebalance_window])
            self.train_full_env = DummyVecEnv([lambda: StockTradingEnv(train_full,
                                                                self.stock_dim,
                                                                self.hmax,
                                                                self.initial_amount,
                                                                self.buy_cost_pct,
                                                                self.sell_cost_pct,
                                                                self.reward_scaling,
                                                                self.state_space,
                                                                self.action_space,
                                                                self.tech_indicator_list,
                                                                print_verbosity=self.print_verbosity)])
            # Model Selection based on sharpe ratio
            if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
                model_use.append('PPO')

                model_ensemble = self.get_model("ppo",self.train_full_env,policy="MlpPolicy",model_kwargs=PPO_model_kwargs)
                model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ppo']) #100_000
            elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
                model_use.append('A2C')

                model_ensemble = self.get_model("a2c",self.train_full_env,policy="MlpPolicy",model_kwargs=A2C_model_kwargs)
                model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['a2c']) #100_000
            else:
                model_use.append('DDPG')

                model_ensemble = self.get_model("ddpg",self.train_full_env,policy="MlpPolicy",model_kwargs=DDPG_model_kwargs)
                model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i), iter_num = i, total_timesteps=timesteps_dict['ddpg']) #50_000

            ############## Training and Validation ends ##############

            ############## Trading starts ##############
            print("======Trading from: ", self.unique_trade_date[i - self.rebalance_window], "to ", self.unique_trade_date[i])
            #print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(model=model_ensemble, name="ensemble",
                                                     last_state=last_state_ensemble, iter_num=i,
                                                     turbulence_threshold = turbulence_threshold,
                                                     initial=initial)
            ############## Trading ends ##############

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame([iteration_list,validation_start_date_list,validation_end_date_list,model_use,a2c_sharpe_list,ppo_sharpe_list,ddpg_sharpe_list]).T
        df_summary.columns = ['Iter','Val Start','Val End','Model Used','A2C Sharpe','PPO Sharpe','DDPG Sharpe']

        return df_summary




