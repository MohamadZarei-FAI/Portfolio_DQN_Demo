import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pdqn_env_v2_1 import Env
from pdqn_agent_v2_1 import Agent

tf.keras.backend.clear_session()

symbols = ['فملی', 'کاما', 'کگل']
env = Env(symbols)

comment = 'V2_100_P1_Step_02'
agent = Agent(env,
              epsilon_decay_range= 1,
              n_episodes= 100,
              learning_rate = 1e-4,
              architecture = (64, 32),
              l2_reg = 5e-4,
              enc1_state = True,
              enc1_test_ratio = 0,
              enc1_training_epochs = 100,
              enc1_l2_reg = 0.001,
              enc2_state = None,
              enc2_test_ratio = 0,
              enc2_training_epochs = 100,
              enc2_l2_reg = 0.001)

agent.ae_1.train_model(comment = comment, plot_training = False, plot_recons = False)
# agent.ae_2.train_model(comment = comment, plot_training = False, plot_recons = False)

agent.train(comment = comment, plot_training = True)

agent.full_evaluation(comment = comment, plot_results = True)