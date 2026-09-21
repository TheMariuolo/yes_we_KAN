import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt




def preprocess(temp : np.array, t_min, t_max, sequence_length = 50, pred_length = 10,train_size = 0.8, batch_size = 32):

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

    return train_loader, val_loader, None, temp_test




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