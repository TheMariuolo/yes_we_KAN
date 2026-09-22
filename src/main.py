import torch  
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import preprocess, train_rnn, train_kan, show_predictions, models_rmse
from class_temperature import LSTM, KAN_temp, FKAN
from sklearn.metrics import root_mean_squared_error

import tomllib
from pathlib import Path



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')






CONFIG_FILE = Path("config.toml")

with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)


SEQUENCE_LENGTH = config['train']['SEQUENCE_LENGTH']
PRED_LENGTH = config['train']['PRED_LENGTH']
BATCH_SIZE = config['train']['BATCH_SIZE']
TRAIN_SIZE = config['train']['TRAIN_SIZE']
HIDDEN_RNN = config['train']['HIDDEN_RNN']
EPOCHS_RNN = config['train']['EPOCHS_RNN']
EPOCHS_KAN = config['train']['EPOCHS_KAN']

path = config['data']['path']






df = pd.read_csv(path, sep=';')
df = df.sort_values('Data', ascending=True).reset_index(drop=True)
temp = df['Temperatura media'].to_numpy()

df['Data'] = pd.to_datetime(df['Data'])
df['doy'] = df['Data'].dt.dayofyear

df = df.sort_values('Data')



climatology = df.groupby('doy')['Temperatura media'].mean()
df['pred_climatology'] = df['doy'].map(climatology)

df['error_climatology'] = root_mean_squared_error(df['Temperatura media'], df['pred_climatology'])

rmse_climatology = df['error_climatology'].mean()







t_min = np.min(temp)
t_max = np.max(temp)



train_loader, val_loader, test_loader, temp_test = preprocess(temp, 
                                                              t_min, 
                                                              t_max, 
                                                              sequence_length=SEQUENCE_LENGTH, 
                                                              pred_length=PRED_LENGTH, 
                                                              train_size=TRAIN_SIZE, 
                                                              batch_size=BATCH_SIZE)






lstm = LSTM(input_size=1, hidden_size=HIDDEN_RNN, num_layers=3, output_size=PRED_LENGTH).to(device)
tot_params_lstm = sum(p.numel() for p in lstm.parameters() if p.requires_grad)
print(f'Total trainable parameters in LSTM: {tot_params_lstm}')



print(f'\nTraining LSTM model. . .\n')
train_rnn(lstm, train_loader, val_loader, num_epochs = 6, learning_rate = 1e-4, device = device)



kan = KAN_temp([SEQUENCE_LENGTH, 80, PRED_LENGTH], grid_size = 9, spline_order = 3, base_activation = nn.SiLU).to(device)
tot_params_kan = sum(p.numel() for p in kan.parameters() if p.requires_grad)
print(f'\nTotal trainable parameters in KAN: {tot_params_kan}')


print(f'\nTraining KAN model. . .\n')
train_kan(kan, train_loader, val_loader, num_epochs = 15, learning_rate = 1e-4, device = device)



fkan = FKAN(features = [SEQUENCE_LENGTH, 100, PRED_LENGTH], gridsize = 3, smooth_initialization=True).to(device)
tot_params_kan = sum(p.numel() for p in fkan.parameters() if p.requires_grad)
print(f'\nTotal trainable parameters in Fourier KAN: {tot_params_kan}')
print(f'\nTraining Fourier KAN model. . .\n')
train_kan(fkan, train_loader, val_loader, num_epochs = 15, learning_rate = 1e-4, device = device, model_name = 'Fourier KAN')




show_predictions(temp_test, t_min, t_max, kan, fkan, lstm, SEQUENCE_LENGTH)

models_rmse(temp_test, t_min, t_max, kan, fkan, lstm, SEQUENCE_LENGTH, rmse_climatology)