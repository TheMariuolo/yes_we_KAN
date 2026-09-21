import torch  
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils import preprocess, train_rnn, train_kan
from class_temperature import LSTM, KAN_temp, FKAN
from sklearn.metrics import root_mean_squared_error



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv('../temperature_bologna.csv', sep=';')
df = df.sort_values('Data', ascending=True).reset_index(drop=True)
temp = df['Temperatura media'].to_numpy()

df['Data'] = pd.to_datetime(df['Data'])
df['doy'] = df['Data'].dt.dayofyear

df = df.sort_values('Data')



climatology = df.groupby('doy')['Temperatura media'].mean()
df['pred_climatology'] = df['doy'].map(climatology)

df['error_climatology'] = root_mean_squared_error(df['Temperatura media'], df['pred_climatology'])

rmse_climatology = df['error_climatology'].mean()



SEQUENCE_LENGTH = 200
PRED_LENGTH = 1
BATCH_SIZE = 16
TRAIN_SIZE = 0.8
HIDDEN_RNN = 50


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





pred_len = 165

with torch.no_grad():
    pred_kan = kan.sample(temp_test, len_sample = pred_len)

with torch.no_grad():
    pred_fkan = fkan.sample(temp_test, len_sample = pred_len)


with torch.no_grad():
    pred_lstm = lstm.sample(temp_test[:SEQUENCE_LENGTH].unsqueeze(0).unsqueeze(-1), len_sample = 300)




pred_kan = (pred_kan+1)/2 * (t_max - t_min) + t_min
pred_fkan = (pred_fkan+1)/2 * (t_max - t_min) + t_min
pred_lstm = (pred_lstm+1)/2 * (t_max - t_min) + t_min

temperature_test = (temp_test+1)/2 * (t_max - t_min) + t_min




plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
plt.plot(pred_lstm.squeeze().detach().numpy(), linewidth=0.6, color='red', label='LSTM')
plt.xlim(150, None)
plt.title('Temperature Prediction with LSTM')
plt.legend()
plt.show()



plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
plt.plot(pred_kan.detach().numpy(), linewidth=0.6, color='red', label='KAN')
plt.xlim(150, None)
plt.title('Temperature Prediction with KAN')
plt.legend()
plt.show()


plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
plt.plot(pred_fkan.detach().numpy(), linewidth=0.6, color='red', label='Fourier KAN')
plt.xlim(150, None)
plt.title('Temperature Prediction with Fourier KAN')
plt.legend()
plt.show()






print('\nCalculating RMSE for different prediction lengths for every model. . .\n')
array_rmse_kan = []
array_rmse_fkan = []
array_rmse_lstm = []


for i in tqdm(range(1, 25)):
    target = []
    pred_temp_kan = []
    pred_temp_fkan = []
    pred_temp_lstm = []

    pred_days = i

    for j in range(SEQUENCE_LENGTH, len(temp_test)-pred_days):
        with torch.no_grad():
            pred_kan = kan.sample(temp_test[j-SEQUENCE_LENGTH:j], len_sample = pred_days)
            pred_fkan = fkan.sample(temp_test[j-SEQUENCE_LENGTH:j], len_sample = pred_days)
            pred_lstm = lstm.sample(temp_test[j-SEQUENCE_LENGTH:j].unsqueeze(0).unsqueeze(-1), len_sample = pred_days)

        target.append(temp_test[j+pred_days])
        pred_temp_kan.append(pred_kan[-1])
        pred_temp_fkan.append(pred_fkan[-1])
        pred_temp_lstm.append(pred_lstm[0][-1])

        

    target, pred_temp_kan, pred_temp_fkan, pred_temp_lstm = np.array(target), np.array(pred_temp_kan), np.array(pred_temp_fkan), np.array(pred_temp_lstm)

    target = (target+1)/2 * (t_max - t_min) + t_min
    pred_temp_kan = (pred_temp_kan+1)/2 * (t_max - t_min) + t_min
    pred_temp_fkan = (pred_temp_fkan+1)/2 * (t_max - t_min) + t_min
    pred_temp_lstm = (pred_temp_lstm+1)/2 * (t_max - t_min) + t_min

    
    
    rmse_kan = root_mean_squared_error(target, pred_temp_kan)
    array_rmse_kan.append(rmse_kan)

    rmse_fkan = root_mean_squared_error(target, pred_temp_fkan)
    array_rmse_fkan.append(rmse_fkan)

    rmse_lstm = root_mean_squared_error(target, pred_temp_lstm)
    array_rmse_lstm.append(rmse_lstm)



plt.plot(array_rmse_kan, label='RMSE KAN')
plt.plot(array_rmse_fkan, label='RMSE Fourier KAN')
plt.plot(array_rmse_lstm, label='RMSE LSTM')
plt.axhline(y=rmse_climatology, color='red', linestyle='--', label=f'RMSE Climatology: {rmse_climatology:.2f}')
plt.ylim(None, rmse_climatology+rmse_climatology*0.4)
plt.legend()
plt.show()