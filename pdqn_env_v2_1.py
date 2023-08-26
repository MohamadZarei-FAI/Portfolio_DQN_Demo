import numpy as np
import pandas as pd
import yfinance as yf

from pathlib import Path
from itertools import product


class Env:
    def __init__(self,
                symbols : tuple,
                data_source : str = 'csv',
                path : str = 'fpy_data/PDQN_P1_',
                normalize : bool = True,
                sampling_beta : float = .3,
                initial_cash : int = 1000000, 
                time_window : int = 5,
                trading_size : int = 10000,
                buy_tc : float = .003712, 
                sell_tc : float = .0088, 
                ):
        self.symbols = tuple(sorted(symbols))
        self.path = path
        self.data_source = data_source
        self.normalize = normalize
        self.sampling_beta = sampling_beta
        self.initial_cash = initial_cash

        # Parameters
        self.time_window = time_window
        # delta
        self.trading_size = trading_size
        # Buying proportioanl trading cost
        self.buy_tc = buy_tc
        # Selling proportional trading cost
        self.sell_tc = sell_tc
        # NAV after taking action
        self.nav = initial_cash
        # NAV at the end of period
        self.nav_end = None
        # Weights vector just after taking action
        self.weights = np.ones((1, 1 + len(self.symbols))) / (1 + len(self.symbols))

        # Weights vector at the end of period
        self.weights_end = None

        # Weights axiliary vector
        self.weights_ = None

        # Actions
        self.actions = np.array(list(product((-1, 0, 1), 
                                repeat = len(self.symbols))), dtype = np.float32)
        
        # NAV vector of all actions
        self.actions_navs = None
        # Wights matrix of all actions
        self.actions_weights = None
        # Auxiliary weights matrix of all actions
        self.actions_weights_ = None

        self.data = None
        self.fund_data = None
        self.prices = None
        self.train = None
        self.test = None

        self.load_data()
        self.preprocess_data()
        self.split_data()
        self.s_probs = self.calc_s_probs()


    @property
    def price_change(self) -> np.ndarray:
        # Calculate price change of assets
        returns = self.data[self.current_step]['Close'].copy()
        returns = returns.sum().values.reshape(-1, 1)
        return returns
    
    @property
    def train_years(self) -> list:
        return [str(year) for year in self.train.index.year.drop_duplicates()]
    
    @staticmethod
    def phi(array : np.ndarray) -> np.ndarray:
        # Add zero as cash price change
        change = np.concatenate([[[0]], array.reshape(-1, 1)], axis = 0)
        return np.exp(change)

    def load_data(self) -> None:
        if self.data_source == 'csv':
            market_data_path = Path(f'{self.path}market_data.csv')
            fund_data_path = Path(f'{self.path}merged_fund_data.csv')
            self.data = pd.read_csv(market_data_path, header = [0, 1], 
                                    index_col= [0], parse_dates= [0])
            self.fund_data = pd.read_csv(fund_data_path, header = [0, 1], 
                                    index_col= [0], parse_dates= [0])
        else:
            self.data = yf.download(self.symbols, start = '2010-01-01', 
                   end = '2017-12-31').drop(columns = 'Close').tz_localize(None)
            self.data.rename(columns = {'Adj Close': 'Close'}, inplace = True)
            self.data.to_csv('data/PDQN_market_data.csv', index = 'Date')
        print("Data is loaded.")

    def preprocess_data(self) -> None:
        self.prices = self.data['Close'].copy()

        # calculate market features
        self.data['Open'] = (self.data['Open'] - self.data['Close'].shift()) / self.data['Close'].shift()
        self.data['High'] = (self.data['Close'] - self.data['High']) / self.data['High']
        self.data['Low'] = (self.data['Close'] - self.data['Low']) / self.data['Low']
        self.data['Close'] = np.log(self.data['Close']).diff()
        self.data['Volume'] = (self.data['Volume'] - self.data['Volume'].shift()) / self.data['Volume'].shift()

        # remove created missing values
        self.data.dropna(inplace = True)

    def split_data(self) -> None:
        # set the last year as test year and the rest of data as train years
        last_year = self.data.index[-1].year
        # split the data to train and test sets
        self.train = self.data.loc[:str(last_year - 1)]
        self.test = self.data.loc[str(last_year)]

    def calc_s_probs(self):
        years = list(self.train.index.year.drop_duplicates())
        test_year = self.test.index[-1].year

        # Truncated Geometric Distribution for sampling
        s_probs = self.sampling_beta * np.power((1 - self.sampling_beta), 
                                                (test_year - np.array(years) - 1))
        s_probs /= 1 - np.power((1 - self.sampling_beta), len(years))

        return s_probs

    def select_episode(self) -> pd.DataFrame:
        years = list(self.train.index.year.drop_duplicates())

        # Sample episode based on defined probabilites
        selected_year = np.random.choice(a = years, p = self.s_probs)
        return self.train.loc[str(selected_year)].copy()

    def get_state(self) -> tuple:
        data = self.data[self.current_step].copy()
        end_date = data.index[-1]

        data = data.T.values
        if data.shape[1] < self.time_window:
            data = np.concatenate([np.zeros((data.shape[0], 
                                            self.time_window - data.shape[1])), data], axis = 1)
        data = np.split(data, data.shape[0]/ len(self.symbols), axis = 0)
        market_features = np.stack(data, axis = 0)

        fund_features_ = self.fund_data.loc[:end_date].iloc[-1].copy().values.reshape(-1, 1)
        fund_features_ = np.split(fund_features_, self.fund_data.shape[1] // len(self.symbols), axis = 0)
        fund_features = np.stack(fund_features_, axis = 0)

        return market_features, fund_features, self.weights_end

    def reset(self, train : bool = True, evaluate : bool = False, year: str = None) -> np.ndarray:
        if train and (not evaluate):
            self.data = self.select_episode()
        elif train and evaluate:
            self.data = self.train.loc[year].copy()
        else:
            self.data = self.test.copy()
        
        offset = len(self.data) % self.time_window
        data_ = self.data.iloc[offset:].copy()
        self.data = [self.data.iloc[:offset]] if offset > 0 else []
        
        self.data += [group for _, group in data_.groupby(np.arange(len(data_)) // self.time_window)]
        
        self.current_step = 0
        self.nav = self.initial_cash
        self.weights = np.ones((1, 1 + len(self.symbols))) / (1 + len(self.symbols))

        change_ratio = self.phi(self.price_change)
        self.nav_end = float(np.dot((self.nav * self.weights), change_ratio))
        self.weights_end = (self.weights * change_ratio.T) / np.dot(self.weights, change_ratio)
        return self.get_state()
    
    def check_feasibility(self) -> tuple:
        # auxiliary variable to maket the code more readable
        trading_fraction = self.trading_size / self.nav_end
        
        # calculate portfolio value decay rate vector in result of taking all actions
        tc_decay_ = self.actions.copy()
        tc_decay_[self.actions == 1] = self.buy_tc
        tc_decay_[self.actions == -1] = self.sell_tc
        tc_decay = trading_fraction * (tc_decay_.sum(axis = 1).reshape(-1, 1))

        # calculate NAV vector just after taking action
        self.actions_navs = self.nav_end * (1 - tc_decay)

        cash_weights_ = self.actions.copy()
        cash_weights_[self.actions == 1] = -1 - self.buy_tc
        cash_weights_[self.actions == -1] = 1 - self.sell_tc
        cash_weights_ = np.sum(cash_weights_, axis = 1).reshape(-1, 1)
        weights_ = self.weights_end.copy().repeat(self.actions.shape[0], axis = 0)
        weights_ += trading_fraction * np.concatenate([cash_weights_, self.actions], axis = 1)
        self.actions_weights_ = weights_.copy()

        # calculate weights just after taking action
        self.actions_weights = weights_ / np.dot(weights_, np.ones((weights_.shape[1], 1)))
       
        # determine feasible actions
        feasibility_ = self.actions_weights.copy()
        feasibility_[:, 0] -= 3 * (trading_fraction + np.max([self.buy_tc, self.sell_tc]))
        feasibility_[:, 1:] -= trading_fraction
        feasibility_matrix = feasibility_ > 0
        feasible_actions = np.all(feasibility_matrix, axis = 1).astype(np.int32)
        return feasibility_matrix, feasible_actions

    def simulate_actions(self) -> tuple:
        # calcualte the change ratio
        change_ratio = self.phi(self.price_change)
        
        # calculate NAV at the end of period
        navs_end = np.dot((self.actions_navs * self.actions_weights), change_ratio)

        # calculate weights at the end of period
        weights_end = (self.actions_weights * change_ratio.T) / np.dot(self.actions_weights, change_ratio)

        # calcuate NAV at the end of period without taking action
        navs_static = np.dot((self.nav_end * self.weights_end), change_ratio)

        # clacualte reward
        rewards = (navs_end - navs_static) / navs_static

        return weights_end, navs_end, rewards


    def step(self, action_idx : int) -> tuple:

        self.current_step += 1
        weights_end, navs_end, rewards = self.simulate_actions()

        self.nav = float(self.actions_navs[action_idx].copy())
        self.nav_end = float(navs_end[action_idx].copy())
        self.weights = self.actions_weights[action_idx].copy().reshape(1, -1)
        self.weights_end = weights_end[action_idx].copy().reshape(1, -1)
        self.weights_ = self.actions_weights_[action_idx].copy().reshape(1, -1)
        
        reward = float(rewards[action_idx].copy())

        
        done = self.current_step == (len(self.data) - 1)
        next_state = self.get_state()
        # next_state = self.get_state() if not done else (np.zeros((5, 3, 20)), self.weights_end)

        return next_state, reward, done, rewards, weights_end
    