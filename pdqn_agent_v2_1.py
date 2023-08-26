import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

import os, random, logging


from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

from IPython.display import clear_output

# persian text correction functions
from arabic_reshaper import reshape
from bidi.algorithm import get_display

from pdqn_encode_v1_1 import AutoEncoder

def reset_random_seeds(seed = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class Agent:
    def __init__(self, env, 
                gamma : float = .9, 
                epsilon_start : float = 1.0, 
                epsilon_min : float = .01, 
                epsilon_exp_decay : bool = True,
                epsilon_decay_range : float = 1,
                learning_rate : float = 1e-4, 
                batch_size : int = 32, 
                update_frequency : int = 10, 
                n_episodes : int = 100,
                architecture : tuple = (128, 64),
                l2_reg : float = 1e-7,
                max_memory : int = 2000,
                enc1_state : bool = True,
                enc1_type : str = 'fc',
                enc1_scale : str = 'standard',
                enc1_hidden_dim : int = 64,
                enc1_latent_dim : int = 20,
                enc1_lookback_period : int = 5,
                enc1_test_ratio : float = 0,
                enc1_batch_size : int = 32,
                enc1_training_epochs : int = 100,
                enc1_dropout : float = .2,
                enc1_l2_reg : float = 1e-3,
                enc1_add_zeros : bool = True,
                enc2_state : bool = True,
                enc2_type : str = 'fc',
                enc2_scale : str = 'standard',
                enc2_hidden_dim : int = 24,
                enc2_latent_dim : int = 12,
                enc2_lookback_period : int = 1,
                enc2_test_ratio : float = 0,
                enc2_batch_size : int = 32,
                enc2_training_epochs : int = 100,
                enc2_dropout : float = .2,
                enc2_l2_reg : float = 5e-4,
                enc2_add_zeros : bool = True,
                vae_recons_loss_weight : int = 1000,
                results_path : str = 'results/'
                ):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_exp_decay = epsilon_exp_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.n_episodes = n_episodes
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.enc1_state = enc1_state
        self.enc1_scale = enc1_scale
        self.enc1_latent_dim = enc1_latent_dim
        self.enc2_state = enc2_state
        self.enc2_scale = enc2_scale
        self.enc2_latent_dim = enc2_latent_dim
        self.results_path = results_path

        if epsilon_exp_decay:
            self.epsilon_decay = round(np.exp(np.log(epsilon_min) / (n_episodes * epsilon_decay_range)), 4)
        else:
            self.epsilon_decay = (epsilon_start - epsilon_min) / (n_episodes * epsilon_decay_range)

        if not self.enc1_state:
            self.enc1_latent_dim = (self.env.train.shape[1] // (len(self.env.symbols))) * enc1_lookback_period
        
        if self.enc2_state is None:
            self.enc2_latent_dim = 0
        elif not self.enc2_state:
            self.enc2_latent_dim = (self.env.fund_data.shape[1] // (len(self.env.symbols))) * enc2_lookback_period
            

        self.state_dim = (len(self.env.symbols) * (self.enc1_latent_dim + self.enc2_latent_dim)) + (len(self.env.symbols) + 1)

        self.actions = self.env.actions
        self.feasible_actions = None
        self.n_actions = np.power(3, len(self.env.symbols))
        self.memory = ReplayMemory(max_memory, self.state_dim, self.n_actions)

        # prepare data to train autoencoder 1
        train_1 = self.env.train.copy().swaplevel(axis = 1).sort_index(axis = 1)
        self.ae_1 = AutoEncoder(train_1, enc1_type, enc1_hidden_dim, enc1_latent_dim,
                                       enc1_lookback_period, enc1_scale,
                                       enc1_test_ratio, enc1_batch_size,
                                       enc1_training_epochs, enc1_dropout, enc1_l2_reg,
                                       enc1_add_zeros, vae_recons_loss_weight, results_path)
        self.enc_1 = self.ae_1.encoder if self.enc1_state else None
        self.scaler_1 = self.ae_1.scaler

        # prepare data to train autoencoder 2
        train_2 = self.env.fund_data.loc[self.env.train.index].copy().swaplevel(axis = 1).sort_index(axis = 1)
        self.ae_2 = AutoEncoder(train_2, enc2_type, enc2_hidden_dim, enc2_latent_dim,
                                       enc2_lookback_period, enc2_scale,
                                       enc2_test_ratio, enc2_batch_size,
                                       enc2_training_epochs, enc2_dropout, enc2_l2_reg,
                                       enc2_add_zeros, vae_recons_loss_weight, results_path)
        self.enc_2 = self.ae_2.encoder if self.enc2_state else None
        self.scaler_2 = self.ae_2.scaler if self.enc2_state else None

        self.idx = tf.range(batch_size * self.n_actions)
        self.train_history = pd.DataFrame(np.zeros((self.n_episodes, 4)), 
                                          columns = ['CR', 'loss', 'r_squared', 'reward'])
        
        reset_random_seeds()
        self.results = {}

        self._log = logging.getLogger(__name__)
        self._log.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(message)s')
        file_handler = logging.FileHandler('data/events.log')
        stream_handler = logging.StreamHandler()
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        self._log.addHandler(file_handler)
        self._log.addHandler(stream_handler)

        self.network = self.build_model()
        self.target_network = self.build_model()
        self.update_target_network()
        
        self.checkpoint = tf.train.Checkpoint(network=self.network)
        self.checkpoint_manager = tf.train.CheckpointManager(
        self.checkpoint, 'data/checkpoints', max_to_keep=1)


    @staticmethod
    def r_squared(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    @staticmethod
    def powerset(array):
        power_set = []
        for i in range(1, 1 << len(array)): 
            power_set.append([array[j] for j in range(len(array)) if (i & (1 << j))])
        return power_set


    def build_model(self) -> tf.keras.Model:
        model = Sequential()
        model.add(InputLayer(input_shape= (self.state_dim,)))
        for dim in self.architecture:
            model.add((Dense(dim, activation= 'selu', kernel_regularizer = l2(l2 = self.l2_reg))))
        model.add(Dense(self.n_actions, activation= 'selu', kernel_regularizer = l2(l2 = self.l2_reg)))

        model.compile(loss='mse', 
                    optimizer= Adam(learning_rate=self.learning_rate),
                    metrics = [self.r_squared])
        return model
    
    def update_target_network(self) -> None:
        self.target_network.set_weights(self.network.get_weights())

    def encode_state(self, state: tuple) -> np.ndarray:
        market_features, fund_features, weights = state

        if self.enc1_scale is not None:
            n_features = int(len(self.env.train.columns) / len(self.env.symbols))
            market_features_ = np.concatenate(market_features, axis = 0).T
            scaled_features = self.scaler_1.transform(market_features_)
            market_features = np.stack(np.split(scaled_features.T, n_features, axis = 0), axis = 0)
        
        market_features = market_features.transpose((1, 2, 0))

        if (self.enc2_scale is not None) and (self.enc2_state is not None):
            n_features = int(len(self.env.fund_data.columns) / len(self.env.symbols))
            fund_features_ = np.concatenate(fund_features, axis = 0).T
            scaled_features = self.scaler_2.transform(fund_features_)
            fund_features = np.stack(np.split(scaled_features.T, n_features, axis = 0), axis = 0)
            fund_features = fund_features.transpose((1, 2, 0))

        if self.enc1_state:
            market_features = self.enc_1.predict(market_features).reshape(1, -1)
        else:
            market_features = market_features.reshape(1, -1)

        if self.enc2_state is None:
            state = np.concatenate([market_features, weights], axis = 1)
            return state
        elif not self.enc2_state:
            fund_features = fund_features.reshape(1, -1)
        else:
            fund_features = self.enc_2.predict(fund_features).reshape(1, -1)
            
        state = np.concatenate([market_features, fund_features, weights], axis = 1)
        return state

    def mapping_rule_1(self, action: int, q_values: np.ndarray) -> int:
        max_q_value = np.NINF
        mapped_action = 13
        action_rules = self.actions[action].copy()
        buying_assets = np.where(action_rules == 1)[0]

        for subset in self.powerset(buying_assets):
            action_ = action_rules.copy()
            action_[subset] = 0
            action_ = np.where((self.actions == action_).all(axis = 1))[0]

            if self.feasible_actions[action_] and q_values[action_] > max_q_value:
                max_q_value = q_values[action_]
                mapped_action = int(action_)
        
        return mapped_action

    def mapping_rule_2(self, action: int, q_values: np.ndarray, feasibility_matrix: np.ndarray) -> int:
        action_rules = self.actions[action].copy()
        infeasible_sellings = np.where(feasibility_matrix[action, 1:] == False)[0]
        action_rules[infeasible_sellings] = 0
        mapped_action = np.where((self.actions == action_rules).all(axis = 1))[0].squeeze()

        if self.feasible_actions[mapped_action] == 0:
            mapped_action = self.mapping_rule_1(mapped_action, q_values)


        return mapped_action

    def mapping_function(self, action: int, q_values: np.ndarray, feasibility_matrix: np.ndarray) -> int:

        if self.feasible_actions[action] == 1:
            return action
        elif not feasibility_matrix[action, 1:].all():
            print('asset shortage')
            mapped_action = self.mapping_rule_2(action, q_values, feasibility_matrix)
        else:
            print('cash shortage')
            mapped_action = self.mapping_rule_1(action, q_values)
        
        return mapped_action

    def act(self, state : tuple) -> int:

        q_values = np.squeeze(self.network.predict(state))
        rand = np.random.rand()
        feasibility_matrix, self.feasible_actions = self.env.check_feasibility()
        
        if rand > self.epsilon:
            selected_action = np.argmax(q_values)
        else:
            feasible_actions = np.where(self.feasible_actions == 1)[0]
            if len(feasible_actions) > 0:
                return np.random.choice(feasible_actions)
            else:
                return int((self.n_actions / 2) + 1)
        
        mapped_action = self.mapping_function(selected_action, q_values, feasibility_matrix)

        return mapped_action

    def remember(self, state : np.ndarray, feasible_actions : np.ndarray, rewards : np.ndarray, 
                next_states : np.ndarray, done : bool) -> None:
        self.memory.store(state, feasible_actions, rewards, next_states, done)

    def replay(self, episode: int) -> None:
        if len(self.memory) < self.batch_size:
            return
        
        states, feasible_actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # Preprocess Batch Data
        next_states = next_states.reshape(-1, self.state_dim)

        next_q_values = self.network.predict(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)
        next_q_values_target = self.target_network.predict(next_states)
        target_q_values = tf.gather_nd(next_q_values_target,
                                    tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1))
        target_q_values = np.array(target_q_values).reshape(self.batch_size, self.n_actions)
        targets = rewards + (1 - dones) * self.gamma * target_q_values
        # q_values = self.network.predict(states)
        # q_values[self.idx, actions] = targets

        loss = self.network.fit(x=states, y=targets, epochs = 1, 
                                batch_size = self.batch_size, shuffle = True)
        
        self.train_history['loss'][episode] += loss.history['loss'][0]
        self.train_history['r_squared'][episode] += loss.history['r_squared'][0]

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            if self.epsilon_exp_decay:
                self.epsilon *= self.epsilon_decay
            else:
                self.epsilon -= self.epsilon_decay

    def plot_train_history(self, comment : str = 'Final'):

        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12))
        self.train_history.plot(subplots=True, linewidth=1, ax=axes, sharex=False, sharey=False)

        titles = self.train_history.columns
        for i, ax in enumerate(axes.flatten()):
            ax.set_title(f'{titles[i]} - last_20pct_mean: {self.train_history.iloc[int(self.n_episodes * .8):].mean().round(4)[i]}')

        fig.suptitle(f'Training History - {comment}', fontsize=18)
        plt.savefig(f'{self.results_path}figure/PDQN_Training_History_{comment}.jpg', dpi = 300)
        plt.show()

    def train(self, comment : str = 'Final', plot_training : bool = True) -> None:
        for episode in range(self.n_episodes):
            # tf.keras.backend.clear_session()
            state_ = self.env.reset()
            state = self.encode_state(state_)
            done = False
            clear_output(wait = True)
            # self._log.debug(f"Episode: {episode + 1}/{self.n_episodes}")
            print(f"Episode: {episode + 1}/{self.n_episodes}")

            while not done:
                action = self.act(state)
                next_state_, reward, done, rewards, next_weights = self.env.step(action)
                next_state = self.encode_state(next_state_)
                next_states = np.concatenate([next_state[0][:-4].reshape(1, -1).repeat(self.n_actions, axis = 0), next_weights], axis = 1)
                
                self.remember(state, self.feasible_actions, rewards, next_states, done)

                self.replay(episode)

                state = next_state

                self.train_history['reward'][episode] += reward    

            self.train_history['CR'][episode] = (self.env.nav_end - self.env.initial_cash) / self.env.initial_cash
            self.update_target_network()
            self.update_epsilon()

        self.train_history.index += 1
        self.train_history.loc[:, ['loss', 'r_squared', 'reward']] /= (252 // self.env.time_window)

        self.train_history.to_csv(f'{self.results_path}data/PDQN_train_history_{comment}.csv', index = True)

        if plot_training:
            self.plot_train_history(comment = comment)
        
    def save_model(self) -> None:
        self.checkpoint_manager.save()

    def load_model(self) -> None:
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)

    def evaluate(self, train : bool = False, year : str = None) -> tuple:
        self.epsilon = 0
        state_ = self.env.reset(train = train, evaluate = True, year = year)
        state = self.encode_state(state_)

        dates = self.env.train.loc[year].index.copy() if train else self.env.test.index.copy()
        # Create the column index
        symbols = list(self.env.symbols)
        weight_levels = ['Cash'] + symbols

        # Create the MultiIndex for the columns
        columns_index = pd.MultiIndex.from_tuples([('creturns', level) for level in symbols] + 
                                                [('navs', level) for level in ['B&H', 'DQN']] + 
                                                [('weights', level) for level in weight_levels] +
                                                [('actions', level) for level in symbols])

        # Create an empty DataFrame
        df = pd.DataFrame(columns = columns_index, index = dates).fillna(0)
        df['creturns'] = self.env.prices.loc[dates] / self.env.prices.loc[dates[0]]

        nav_bh = self.env.nav
        weights_bh = self.env.weights.copy()
        done = False

        for i in range(len(self.env.data)):

            dates_ = self.env.data[i].index.copy()
            price_changes = self.env.data[i]['Close'].copy()
            nav = self.env.nav
            weights = self.env.weights.copy()
            
            for j in range(len(dates_)):
                change_ratio = self.env.phi(price_changes.loc[dates_[j]].values)
                nav = float(np.dot(nav * weights, change_ratio))
                nav_bh = float(np.dot(nav_bh * weights_bh, change_ratio))
                weights = (weights * change_ratio.T) / np.dot(weights, change_ratio)
                weights_bh = (weights_bh * change_ratio.T) / np.dot(weights_bh, change_ratio)
                df.loc[dates_[j], 'navs'] = [nav_bh, nav]
                df.loc[dates_[j], 'weights'] = weights.squeeze()
            
            action = self.act(state)
            df.loc[dates_[-1], 'actions'] = self.actions[action]

            if done:
                break
            
            next_state_, _, done, _, _ = self.env.step(action)
            next_state = self.encode_state(next_state_)
            state = next_state

        df['navs'] /= df['navs'].iloc[0]


        # Cumulative Return
        CR = (df['navs'].iloc[-1] - df['navs'].iloc[0]) / df['navs'].iloc[0]
        CR = CR.round(4)

        # Sharpe Ratio
        navs_returns = np.log(df['navs']).diff()
        SR = (navs_returns.mean() / navs_returns.std()) * np.sqrt(252)
        SR = SR.round(4)

        # Sterling Ratio
        negative_returns = np.minimum(navs_returns, np.zeros_like(navs_returns))
        SterR = (navs_returns.mean() / np.sqrt(np.power(negative_returns, 2).mean())) * np.sqrt(252)
        SterR = SterR.round(4)

        metrics = pd.concat([CR, SR, SterR], keys = ['CR', 'SR', 'SterR'], axis = 1)
        metrics.loc['outperf'] = metrics.loc['DQN'] - metrics.loc['B&H']

        return df, metrics

    def load_results(self, comment : str = 'final'):

        try:
            train_results = pd.read_csv(f'{self.results_path}data/PDQN_train_results_{comment}.csv', 
                                        header = [0,1], index_col = 0, parse_dates = [0])
            train_metrics = pd.read_csv(f'{self.results_path}data/PDQN_train_metrics_{comment}.csv', 
                                        header = 0, index_col = [0, 1])
            test_results = pd.read_csv(f'{self.results_path}data/PDQN_test_results_{comment}.csv', 
                                        header = [0,1], index_col = 0, parse_dates = [0])
            test_metrics = pd.read_csv(f'{self.results_path}data/PDQN_test_metrics_{comment}.csv', 
                                        header = 0, index_col = 0)
            for year in self.env.train_years:
                self.results[f'train_{year}'] = (train_results.loc[year], train_metrics.loc[year])
            self.results['test'] = (test_results, test_metrics)
            print('results are loaded')

        except Exception as e:
            print(e)

    def plot_evaluation_results(self, train_outperf : float, test_outperf : float, comment : str = 'Final'):
        set_names = list(self.results.keys())
        symbol_colors = ['orange', 'blue', 'brown']

        for i in range(len(set_names) // 3):
            fig, axs = plt.subplots(nrows = 3, ncols = 3, figsize = (18, 14))
            for r in range(3):
                set_name = list(self.results.keys())[3 * i + r]
                results = self.results[set_name]
                navs = results[0].loc[:, 'navs']
                weights = results[0].loc[:, 'weights']
                creturns = results[0].loc[:, 'creturns']
                actions = results[0].loc[:, 'actions']
                CR_outperf = results[1].loc['outperf', 'CR']

                ax_navs = axs[0, r]
                ax_navs.plot(navs)
                ax_navs.set_title(f'{set_name} NAVs, CR outperf: {round(CR_outperf *100, 2)}')
                ax_navs.legend([get_display(reshape(text)) for text in navs.columns])
                ax_navs.grid()

                ax_weights = axs[1, r]
                ax_weights.plot(weights)
                ax_weights.set_title(f'{set_name} Weights')
                ax_weights.legend([get_display(reshape(text)) for text in weights.columns])
                ax_weights.grid()

                ax_actions = axs[2, r]
                for idx, column in enumerate(creturns.columns):
                    ax_actions.plot(creturns.index, creturns[column], label=get_display(reshape(column)))
                
                # place markers based on decisions
                colors = {1: 'g', -1: 'r'}  # 1 for buy(green), -1 for sell(red), 0 for hold(blue)
                markers = {1: '^', -1: 'v'}  # different markers for buy, sell, hold
                for column in actions.columns:
                    for decision in [-1, 1]:
                        mask = actions[column] == decision
                        ax_actions.scatter(creturns.loc[mask].index, 
                        creturns.loc[mask, column], 
                        color=colors[decision], 
                        marker=markers[decision], 
                        # label=f'{column}_{decision}',
                        s = 40,
                        alpha = .8)

                ax_actions.set_title(f'{set_name} Actions')
                ax_actions.legend([get_display(reshape(text)) for text in creturns.columns])
                ax_actions.grid()

            fig.suptitle(f'Evaluation Results - {comment} train OP: {round(train_outperf *100, 2)}, test OP: {round(test_outperf *100, 2)} - {i + 1}', fontsize=18)
            plt.savefig(f'{self.results_path}figure/PDQN_Evaluation_History_{comment}_{i + 1}.jpg', dpi = 300)
            plt.tight_layout()
            plt.show()


    def full_evaluation(self, comment : str = 'Final', 
                load_from_csv : bool = False, 
                plot_results : bool = True) -> None:
    
        if not load_from_csv:
            
            train_results = []
            train_metrics = []

            for year in self.env.train_years:
                self.results[f'train_{year}'] = self.evaluate(train = True, year = year)
                train_results.append(self.results[f'train_{year}'][0])
                train_metrics.append(self.results[f'train_{year}'][1])

            train_results = pd.concat(train_results, axis = 0)
            train_metrics = pd.concat(train_metrics, axis = 0, keys = self.env.train_years)

            self.results['test']  = self.evaluate(train = False)
            # save results
            train_results.to_csv(f'{self.results_path}data/PDQN_train_results_{comment}.csv', index = True)
            train_metrics.to_csv(f'{self.results_path}data/PDQN_train_metrics_{comment}.csv', index = True)
            self.results['test'][0].to_csv(f'{self.results_path}data/PDQN_test_results_{comment}.csv', index = True)
            self.results['test'][1].to_csv(f'{self.results_path}data/PDQN_test_metrics_{comment}.csv', index = True)
        
        else:
            self.load_results(comment = comment)

        train_outperfs = [self.results[f'train_{year}'][1].loc['outperf', 'CR'].copy() for year in self.env.train_years]
        train_outperf = np.sum(np.multiply(train_outperfs, self.env.s_probs))
        test_outperf = self.results['test'][1].loc['outperf', 'CR'].copy()

        print(f'train_outperfs by year:')
        for i, outperf in enumerate(train_outperfs):
            print(f'{self.env.train_years[i]}: {round(train_outperfs[i] *100, 2)}')
        print(f'Average Train Outperf: {round(train_outperf *100, 2)}')
        print(f'test outperf: {round(test_outperf *100, 2)}')

        if plot_results:
            self.plot_evaluation_results(train_outperf = train_outperf, 
                                        test_outperf = test_outperf, 
                                        comment = comment)


class ReplayMemory:
    def __init__(self, mem_size : int, 
                 state_dim : tuple, 
                 n_actions : int, 
                ):
        self.mem_size = mem_size
        self.mem_counter = 0

        self.state_memory = np.zeros((self.mem_size, state_dim), dtype=  np.float32)
        self.feasible_actions_memory = np.zeros((self.mem_size, n_actions), dtype=  np.int32)
        self.rewards_memory = np.zeros((self.mem_size, n_actions), dtype = np.float32)
        self.new_states_memory = np.zeros((self.mem_size, n_actions, state_dim), dtype = np.float32)
        self.done_memory = np.zeros((self.mem_size, 1), dtype = bool)
        

    def store(self, state, feasible_actions, rewards, new_states, done):
        index = self.mem_counter % self.mem_size

        self.state_memory[index] = state
        self.feasible_actions_memory[index] = feasible_actions.reshape(1, -1)
        self.rewards_memory[index] = rewards.reshape(1, -1)
        self.new_states_memory[index] = new_states
        self.done_memory[index] = done

        self.mem_counter += 1

    def sample(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        state = self.state_memory[batch]
        feasible_actions = self.feasible_actions_memory[batch]
        rewards = self.rewards_memory[batch]
        new_states = self.new_states_memory[batch]
        dones = self.done_memory[batch].reshape(-1, 1)
        
        return state, feasible_actions, rewards, new_states, dones

    def __len__(self) -> int:
        return min(self.mem_counter, self.mem_size)
    