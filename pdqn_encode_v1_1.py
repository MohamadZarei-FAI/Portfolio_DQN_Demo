import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os, random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import r2_score
from keras.layers import InputLayer, Dense, LSTM, GRU, Conv1D, Conv1DTranspose
from keras.layers import Layer, TimeDistributed, Dropout, Reshape, Flatten, RepeatVector
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.metrics import MeanSquaredError
from keras import backend as K

from IPython.display import clear_output

# persian text correction functions
from arabic_reshaper import reshape
from bidi.algorithm import get_display

def reset_random_seeds(seed = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'


class AutoEncoder:
    def __init__(self, data,
                enc_type = 'fc',
                hidden_dim = 32,
                latent_dim = 20,
                lookback_preiod = 5,
                scale = None,
                test_ratio = 0,
                batch_size = 32,
                epochs = 100,
                dropout = .2,
                l2_reg = 5e-4,
                add_zeros = True,
                recons_loss_weight = 1000,
                results_path : str = 'results/'
                ):
        self.data = data
        self.scale = scale
        self.lookback_period = lookback_preiod
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.epochs = epochs
        self.add_zeros = add_zeros
        self.results_path = results_path

        reset_random_seeds()

        if len(self.data.columns[0]) > 1:
            dataset_names = self.data.columns.get_level_values(level = 0).drop_duplicates()
            self.n_datasets = len(dataset_names)
            self.series_names = list(self.data[dataset_names[0]].columns)
        else:
            self.n_datasets = 1
            self.seris_names = list(self.data.columns)

        if enc_type == 'fc':
            self.autoencoder = FCAutoEncoder(input_dim = len(self.series_names), 
                                            hidden_dim = hidden_dim, 
                                            latent_dim = latent_dim, 
                                            lookback_period = lookback_preiod,
                                            l2_reg = l2_reg)
            self.autoencoder.compile(optimizer = Adam(), loss = 'mse', metrics = [self.r_squared])
        elif enc_type == 'lstm':
            self.autoencoder = LSTMAutoEncoder(input_dim = len(self.series_names), 
                                            hidden_dim = hidden_dim, 
                                            latent_dim = latent_dim, 
                                            lookback_period = lookback_preiod,
                                            dropout = dropout)
            self.autoencoder.compile(optimizer = Adam(), loss = 'mse', metrics = [self.r_squared])
        elif enc_type == 'gru':
            self.autoencoder = GRUAutoEncoder(input_dim = len(self.series_names), 
                                            hidden_dim = hidden_dim, 
                                            latent_dim = latent_dim, 
                                            lookback_period = lookback_preiod,
                                            dropout = dropout)
            self.autoencoder.compile(optimizer = Adam(), loss = 'mse', metrics = [self.r_squared])
        elif enc_type == 'conv':
            self.autoencoder = ConvAutoEncoder(input_dim = len(self.series_names), 
                                            hidden_dim = hidden_dim, 
                                            latent_dim = latent_dim, 
                                            lookback_period = lookback_preiod,
                                            l2_reg = l2_reg)
            self.autoencoder.compile(optimizer = Adam(), loss = 'mse', metrics = [self.r_squared])
        elif enc_type == 'fcvae':
            self.autoencoder = FCVAE(input_dim = len(self.series_names), 
                                            hidden_dim = hidden_dim, 
                                            latent_dim = latent_dim, 
                                            lookback_period = lookback_preiod,
                                            l2_reg = l2_reg, 
                                            recons_loss_weight = recons_loss_weight)
            self.autoencoder.compile(optimizer = Adam(), 
                                    loss = self.autoencoder.vae_loss, 
                                    metrics = [self.r_squared])
            

        self.encoder = self.autoencoder.encoder
        
        
        self.train_sets = {}
        self.set_names = None
        self.results = {}
        self.scaler = self.scale_data() if self.scale is not None else None
        self.preprocess_data()

    @staticmethod
    def r_squared(y_true, y_pred):
        SS_res =  K.sum(K.square( y_true-y_pred )) 
        SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) ) 
        return ( 1 - SS_res/(SS_tot + K.epsilon()) )

    @staticmethod
    def r_square(y_true, y_pred):
        error = y_true - y_pred
        bias = y_true - y_true.mean()
        sse = np.power(error, 2).sum().sum()
        sst = np.power(bias,  2).sum().sum()
        r_square = 1 - (sse / sst)
        return max(0, float(r_square))

    def split_data(self, data = None):
        data_ = self.data.copy() if data is None else data.copy()
        train , test = train_test_split(data_, test_size = self.test_ratio, shuffle = False)
        return train, test
    
    def scale_data(self):
        if self.test_ratio > 0:
            data, _ = self.split_data()
        else:
            data = self.data.copy()

        if self.scale == 'standard':
            scaler = StandardScaler()
        elif self.scale == 'minmax':
            scaler = MinMaxScaler()

        if self.add_zeros:
            data = np.concatenate([np.zeros((self.lookback_period -1, data.shape[1])), data], axis = 0)
        
        scaler.fit(data)
        self.data = pd.DataFrame(scaler.transform(self.data.copy()),
                                columns = self.data.columns,
                                index = self.data.index)              
        return scaler

    def preprocess_data(self):
        data = self.data.values.copy()

        if self.add_zeros:
            data = np.concatenate([np.zeros((self.lookback_period -1, data.shape[1])), data], axis = 0)
        
        data = np.array([data[j : j + self.lookback_period] for j in range(len(data) - self.lookback_period)])
        
        if self.test_ratio > 0:
            train , test = self.split_data(data)
            self.train_sets['train'] = train
            self.train_sets['test'] = test
            self.set_names = ['train', 'test']
        else:
            self.train_sets['train'] = data
            self.set_names = ['train']

        if self.n_datasets > 1:
            for set_name, X in self.train_sets.items():
                self.train_sets[set_name] = np.concatenate(np.split(X, self.n_datasets, axis = 2), axis = 0)

    def train_model(self, comment : str = 'Final', 
                    plot_training : bool = True, 
                    plot_recons : bool = True):
        
        X_train = self.train_sets['train']

        if self.test_ratio > 0:
            X_test = self.train_sets['test']
            history = self.autoencoder.fit(X_train, X_train, epochs = self.epochs, 
                                        batch_size = self.batch_size, 
                                        validation_data = (X_test, X_test),
                                        shuffle = True)
        else:
            history = self.autoencoder.fit(X_train, X_train, epochs = self.epochs, 
                                        batch_size = self.batch_size,
                                        shuffle = True)
        
        self.evaluate()
        
        if plot_training:
            plt.figure(figsize = (12, 9))
            plt.plot(history.history['loss'], label='train')
            plt.plot(history.history['val_loss'], label='test')
            plt.legend(fontsize = 15)
            plt.xlabel('Epoch', fontsize = 15)
            plt.ylabel('Loss', fontsize = 15)
            plt.title(f'Autoencoder Training loss - {comment}', fontsize = 18)
            plt.savefig(f'{self.results_path}figure/PDQN_Autoencoder_loss_{comment}.jpg', dpi = 300)
            plt.show()

        if plot_recons:
            self.plot_reconstruction(comment)

    def reconstruct(self, data):
        data = np.array([data[i][-1] for i in range(len(data))])

        if self.scale is not None:
            if self.n_datasets > 1:
                data = np.concatenate(np.split(data, self.n_datasets, axis = 0), axis = 1)

            data = self.scaler.inverse_transform(data)

            if self.n_datasets > 1:
                data = np.concatenate(np.split(data, self.n_datasets, axis = 1), axis = 0)

        data = pd.DataFrame(data, columns = self.series_names)

        return data

    def evaluate(self):

        for set_name, X in self.train_sets.items():
            results = self.autoencoder.evaluate(X, X)

            X_pred = self.autoencoder.predict(X)

            X = self.reconstruct(X)
            X_recons = self.reconstruct(X_pred)

            r_squared_recons = r2_score(X, X_recons)

            self.train_sets[set_name] = X
            self.results[set_name] = X_recons.copy(), results[1], r_squared_recons

        clear_output(wait = True)
        for set_name, result in self.results.items():
            print(f'{set_name} R-Squared(pct):     {round(result[1] *100, 1)}')
            print(f'{set_name} R-Squared-recons(pct): {round(result[2] *100, 1)}')
            
            
    def plot_reconstruction(self, comment : str = 'Final'):

        for plot_idx in range(len(self.set_names)):
            set_name = self.set_names[plot_idx]
            y = self.train_sets[set_name]
            reconstructed = self.results[set_name][0]
            r_squared = self.results[set_name][1]
            r_squared_recons = self.results[set_name][2]

            n_rows = len(self.series_names) // 3 + (len(self.series_names) % 3 != 0)
            fig, axs = plt.subplots(n_rows, 3, figsize=(18 , 6 * n_rows))
            for i in range(len(self.series_names)):
                ax = axs[np.unravel_index(i, (n_rows, 3))]
                ax.plot(y[self.series_names[i]], label='Initial')
                ax.plot(reconstructed[self.series_names[i]], label='Reconstructed')
                ax.set_title('Serie: {}'.format(get_display(reshape(self.series_names[i]))), fontsize = 12)
                ax.legend()
            # plt.tight_layout()
            fig.suptitle(f'Autoencoder Reconstruction Plots - {set_name} - R-Squared(pct): {round(r_squared * 100, 1)} - R-Square-recons(pct): {round(r_squared_recons * 100, 1)} - {comment}', fontsize = 18)
            plt.savefig(f'{self.results_path}figure/PDQN_Autoencoder_evaluation_{comment}_{1 + plot_idx}_{set_name}.jpg', dpi = 300)
            plt.show()



class FCAutoEncoder(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                l2_reg = 1e-7):
        super(FCAutoEncoder, self).__init__()
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            Flatten(),
            Dense(units = hidden_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'encoder_1'),
            Dense(units = latent_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'encoder_2')
        ])

        self.decoder = Sequential([
            InputLayer(input_shape = (latent_dim,)),
            Dense(units = latent_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'decoder_1'),
            Dense(units = hidden_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'decoder_2'),
            Dense(units = input_dim * lookback_period, activation= 'selu'),
            Reshape((lookback_period, input_dim))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class LSTMAutoEncoder(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                dropout):
        super(LSTMAutoEncoder, self).__init__()
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            LSTM(units = hidden_dim, return_sequences = True, dropout = dropout, name = 'encoder_1'),
            LSTM(units = latent_dim, return_sequences = False, dropout = dropout, name = 'encoder_2')
        ])

        self.decoder = Sequential([
            InputLayer(input_shape = (latent_dim,)),
            RepeatVector(n = lookback_period, name = 'encoder_decoder_bridge'),
            LSTM(units = latent_dim, return_sequences = True, dropout = dropout, name = 'decoder_1'),
            LSTM(units = hidden_dim, return_sequences = True, dropout = dropout, name = 'decoder_2'),
            TimeDistributed(Dense(units = input_dim))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class GRUAutoEncoder(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                dropout):
        super(GRUAutoEncoder, self).__init__()
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            GRU(units = hidden_dim, return_sequences = True, dropout = dropout, name = 'encoder_1'),
            GRU(units = latent_dim, return_sequences = False, dropout = dropout, name = 'encoder_2')
        ])

        self.decoder = Sequential([
            InputLayer(input_shape = (latent_dim,)),
            RepeatVector(n = lookback_period, name = 'encoder_decoder_bridge'),
            GRU(units = latent_dim, return_sequences = True, dropout = dropout, name = 'decoder_1'),
            GRU(units = hidden_dim, return_sequences = True, dropout = dropout, name = 'decoder_2'),
            TimeDistributed(Dense(units = input_dim))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class ConvAutoEncoder(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                num_filters = 32,
                kernel_size = 3,
                l2_reg = 1e-7):
        super(ConvAutoEncoder, self).__init__()
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            Conv1D(filters=num_filters, kernel_size=kernel_size, activation='selu', padding='same', kernel_regularizer=l2(l2_reg)),
            Flatten(),
            Dense(units=hidden_dim, activation='selu', kernel_regularizer=l2(l2_reg)),
            Dense(units=latent_dim, activation='selu', kernel_regularizer=l2(l2_reg))
        ])

        self.decoder = Sequential([
            InputLayer(input_shape=(latent_dim,)),
            Dense(units=hidden_dim, activation='selu', kernel_regularizer=l2(l2_reg)),
            Dense(units=num_filters * lookback_period, activation='selu'),
            Reshape((lookback_period, num_filters)),
            Conv1DTranspose(filters=input_dim, kernel_size=kernel_size, activation='selu', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    

class Conv2AutoEncoder(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                num_filters = 32,
                kernel_size = 3,
                l2_reg = 1e-7):
        super(Conv2AutoEncoder, self).__init__()
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            Conv1D(filters=num_filters, kernel_size=kernel_size, activation='selu', padding='same', kernel_regularizer=l2(l2_reg)),
            Flatten(),
            Dense(units=hidden_dim, activation='selu', kernel_regularizer=l2(l2_reg)),
            Dense(units=latent_dim, activation='selu', kernel_regularizer=l2(l2_reg))
        ])

        self.decoder = Sequential([
            InputLayer(input_shape=(latent_dim,)),
            Dense(units=hidden_dim, activation='selu', kernel_regularizer=l2(l2_reg)),
            Dense(units=num_filters * lookback_period, activation='selu'),
            Reshape((lookback_period, num_filters)),
            Conv1DTranspose(filters=input_dim, kernel_size=kernel_size, activation='selu', padding='same')
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded



class FCVAE(Model):
    def __init__(self,
                input_dim,
                hidden_dim,
                latent_dim,
                lookback_period,
                l2_reg = 1e-7,
                recons_loss_weight = 1000):
        super(FCVAE, self).__init__()

        self.recons_loss_weight = recons_loss_weight
        
        self.encoder = Sequential([
            InputLayer(input_shape = (lookback_period, input_dim)),
            Flatten(),
            Dense(units = hidden_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'encoder_1'),
            Dense(units = 2 * latent_dim, kernel_regularizer = l2(l2 = l2_reg), name = 'encoder_2')
        ])

        self.decoder = Sequential([
            InputLayer(input_shape = (latent_dim,)),
            Dense(units = latent_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'decoder_1'),
            Dense(units = hidden_dim, activation= 'selu', kernel_regularizer = l2(l2 = l2_reg), name = 'decoder_2'),
            Dense(units = input_dim * lookback_period, activation= 'selu'),
            Reshape((lookback_period, input_dim))
        ])

    def reconstruction_loss(self, y, y_pred):
        error = y - y_pred
        reconstruction_loss = K.mean(K.square(error), axis = [1, 2])
        return reconstruction_loss
    
    def kl_loss(self, y, y_pred):
        kl_loss = -.5 * K.sum(1 + self.log_var - K.square(self.mean) - K.exp(self.log_var), axis = 1)
        return kl_loss

    def vae_loss(self, y, y_pred):
        reconstruction_loss = self.reconstruction_loss(y, y_pred)
        kl_loss = self.kl_loss(y, y_pred)
        vae_loss = self.recons_loss_weight * reconstruction_loss + kl_loss
        return vae_loss

    def call(self, x):
        encoded = self.encoder(x)
        mean, log_var = tf.split(encoded, num_or_size_splits = 2, axis = 1)
        self.mean = mean
        self.log_var = log_var
        z = Sampling()([mean, log_var])
        decoded = self.decoder(z)
        return decoded
    
    def encode(self, x):
        encoded = self.encoder(x)
        mean, log_var = tf.split(encoded, num_or_size_splits = 2, axis = 1)
        return mean


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
