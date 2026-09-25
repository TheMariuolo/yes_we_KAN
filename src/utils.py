import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import root_mean_squared_error, r2_score




def preprocess(temp : np.array, t_min, t_max, sequence_length = 50, train_size = 0.8, batch_size = 32):

    if isinstance(temp, list):
        raise TypeError("temp argument must be a 1D np.array, not a list")
    
    if train_size >= 1.0 or train_size <= 0.0:
        raise ValueError("train_size argument must be a float between 0 and 1")

    temp_norm = 2*(temp - t_min) / (t_max - t_min) -1

    temp_train = temp_norm[:-365]
    temp_test = temp_norm[-365:]

    grouped_temp = [[temp_train[i:i+sequence_length], temp_train[i+sequence_length]] for i in range(len(temp_train)-sequence_length)]
   
    X = np.stack([np.asarray(x) for x, _ in grouped_temp])   
    y = np.array([y for _, y in grouped_temp])


    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)


    train_size = int(train_size * len(X_t))

    X_train = X_t[:train_size]
    y_train = y_t[:train_size]

    X_val = X_t[train_size:]
    y_val = y_t[train_size:]


    dataset_train = TensorDataset(X_train.unsqueeze(-1), y_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0)
    print('Len train loader:', len(train_loader)*batch_size)

    dataset_val = TensorDataset(X_val.unsqueeze(-1), y_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=0)
    print('Len val loader:', len(val_loader)*batch_size)
        
   
    temp_test = torch.tensor(temp_test, dtype=torch.float32)

    return train_loader, val_loader, temp_test




# --------------------------------------------------------------------------------------------------------------------- #


def train_rnn(rnn, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device = 'cpu'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        rnn.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = rnn(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
           

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        rnn.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                val_outputs = rnn(val_inputs)
                val_loss = criterion(val_outputs.squeeze(), val_targets)

                val_running_loss += val_loss.item() * val_inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM Training')
    plt.legend()
    plt.show()



# --------------------------------------------------------------------------------------------------------------------- #



def train_kan(kan, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device = 'cpu', model_name = 'KAN'):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(kan.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs)):
        kan.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = kan(inputs.squeeze(-1))
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
           

        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)

        kan.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for val_inputs, val_targets in val_loader:
                val_inputs, val_targets = val_inputs.to(device), val_targets.to(device)

                val_outputs = kan(val_inputs.squeeze(-1))
                val_loss = criterion(val_outputs.squeeze(), val_targets)

                val_running_loss += val_loss.item() * val_inputs.size(0)

        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Val Loss: {val_epoch_loss:.4f}')

    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name} Training')
    plt.legend()
    plt.show()



def show_predictions(temp_test, t_min, t_max, kan, fkan, lstm, sequence_length):

    pred_len = len(temp_test) - sequence_length

    with torch.no_grad():
        pred_kan = kan.sample(temp_test, len_sample = pred_len)
        pred_fkan = fkan.sample(temp_test, len_sample = pred_len)
        pred_lstm = lstm.sample(temp_test[:sequence_length].unsqueeze(0).unsqueeze(-1), len_sample = pred_len)




    pred_kan = (pred_kan+1)/2 * (t_max - t_min) + t_min
    pred_fkan = (pred_fkan+1)/2 * (t_max - t_min) + t_min
    pred_lstm = (pred_lstm+1)/2 * (t_max - t_min) + t_min

    temperature_test = (temp_test+1)/2 * (t_max - t_min) + t_min

    r2_lstm = r2_score(temperature_test[sequence_length:].detach().numpy(), pred_lstm[0][sequence_length:].squeeze().detach().numpy())
    r2_kan = r2_score(temperature_test[sequence_length:].detach().numpy(), pred_kan[sequence_length:].detach().numpy())
    r2_fkan = r2_score(temperature_test[sequence_length:].detach().numpy(), pred_fkan[sequence_length:].detach().numpy())

    plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
    plt.plot(pred_lstm.squeeze().detach().numpy(), linewidth=0.6, color='red', label='LSTM')
    plt.xlim(150, None)
    plt.title('Temperature Prediction with LSTM')

    handles, labels = plt.gca().get_legend_handles_labels()
    empty_patch = mpatches.Patch(color="none", label=rf"$R^2 = {r2_lstm:.3f}$")
    handles.append(empty_patch)
    labels.append(rf"$R^2 = {r2_lstm:.3f}$")

    plt.legend(handles=handles, labels=labels, loc="upper right")
    plt.show()


    

    plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
    plt.plot(pred_kan.detach().numpy(), linewidth=0.6, color='red', label='KAN')
    plt.xlim(150, None)
    plt.title('Temperature Prediction with KAN')

    handles, labels = plt.gca().get_legend_handles_labels()
    empty_patch = mpatches.Patch(color="none", label=rf"$R^2 = {r2_kan:.3f}$")
    handles.append(empty_patch)
    labels.append(rf"$R^2 = {r2_kan:.3f}$")

    plt.legend(handles=handles, labels=labels, loc="upper right")
    plt.show()


    plt.plot(temperature_test.detach().numpy(), linewidth=0.6, color='blue', label='Ground Truth')
    plt.plot(pred_fkan.detach().numpy(), linewidth=0.6, color='red', label='Fourier KAN')
    plt.xlim(150, None)
    plt.title('Temperature Prediction with Fourier KAN')

    handles, labels = plt.gca().get_legend_handles_labels()
    empty_patch = mpatches.Patch(color="none", label=rf"$R^2 = {r2_fkan:.3f}$")
    handles.append(empty_patch)
    labels.append(rf"$R^2 = {r2_fkan:.3f}$")

    plt.legend(handles=handles, labels=labels, loc="upper right")
    plt.show()



def models_rmse(temp_test, t_min, t_max, kan, fkan, lstm, sequence_length, rmse_climatology):

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

        for j in range(sequence_length, len(temp_test)-pred_days):
            with torch.no_grad():
                pred_kan = kan.sample(temp_test[j-sequence_length:j], len_sample = pred_days)
                pred_fkan = fkan.sample(temp_test[j-sequence_length:j], len_sample = pred_days)
                pred_lstm = lstm.sample(temp_test[j-sequence_length:j].unsqueeze(0).unsqueeze(-1), len_sample = pred_days)

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