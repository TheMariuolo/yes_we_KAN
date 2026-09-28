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

from tabulate import tabulate



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



CONFIG_FILE = Path("config.example.toml")

with open(CONFIG_FILE, "rb") as f:
        config = tomllib.load(f)




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


temp_norm = 2*(temp - t_min) / (t_max - t_min) -1

temp_norm = torch.tensor(temp_norm, dtype=torch.float32)




checkpoint_kan = torch.load("pretrained/kan_pretrained.pth", map_location=device, weights_only=True)
checkpoint_fkan = torch.load("pretrained/fkan_pretrained.pth", map_location=device, weights_only=True)


kan = KAN_temp([200, 80, 1], grid_size = 9, spline_order = 3, base_activation = nn.SiLU).to(device)
fkan = FKAN(features = [200, 80, 1], gridsize = 3, smooth_initialization=True).to(device)

kan.load_state_dict(checkpoint_kan)
fkan.load_state_dict(checkpoint_fkan)


with torch.no_grad():
        pred_kan = kan.sample(temp_norm[-200:], len_sample = 10)
        pred_fkan = fkan.sample(temp_norm[-200:], len_sample = 10)




temp_norm = (temp_norm+1)/2 * (t_max - t_min) + t_min
pred_kan = (pred_kan+1)/2 * (t_max - t_min) + t_min
pred_fkan = (pred_fkan+1)/2 * (t_max - t_min) + t_min


plt.plot(temp_norm[-10:], label = "Ground Truth", color = "blue", linewidth = 2)
plt.plot(range(9,20), pred_kan[-11:], label = "KAN Prediction", color = "red", linestyle = "--")
plt.plot(range(9,20), pred_fkan[-11:], label = "FKAN Prediction", color = "green", linestyle = "-.")
plt.title("Temperature Forecasting")
plt.xlabel("Time [Days]")
plt.ylabel("Temperature [°C]")
plt.legend()
plt.show()




dates = pd.date_range(start=(df['Data'].iloc[-1]+pd.Timedelta(days=1)), periods=10, freq='D').strftime("%Y-%m-%d")



df = pd.DataFrame(
    {
        "Date": dates,
        "Prediction KAN (°C)": pred_kan[-10:].detach().numpy(),
        "Prediction FKAN (°C)": pred_fkan[-10:].detach().numpy()
    }
)



print(
    tabulate(df, headers="keys", tablefmt="rounded_outline", showindex=False)
)