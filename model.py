import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

class Linear_QNet(nn.Module):
    '''This model's outputs reflect an understanding of the expected discounted future
    reward of each of the possible actions that can be taken in this state. The
    largest outputted value corresponds to the most profitable action from the
    inputted state. '''

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        new_state = state
        # for layer in self.layers:
        #     new_state = layer(new_state)
        #     new_state = self.relu(new_state)
        #     new_state = self.dropout(new_state)
        # new_state = self.linear(new_state)

        # old code
        new_state = self.linear1(state)
        new_state = self.relu(new_state)
        # new_state = self.dropout(new_state)
        new_state = self.linear2(new_state)
        
        return new_state
    
    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        os.makedirs(model_folder_path, exist_ok=True)

        model_file_path = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), model_file_path)

class QTrainer:
    def __init__(self, model: nn.Module, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predict Q value with current state(s)
        q_values = self.model(state) # Q values for `state`

        target = q_values.clone()

        for idx in range(len(state)):
            if done[idx]:
                q_new = reward[idx]
            else:
                next_state_q = self.model(next_state[idx])
                q_new = reward[idx] + self.gamma * torch.max(next_state_q)

            target[idx][torch.argmax(action[idx]).item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, q_values)
        loss.backward()
        self.optimizer.step()
