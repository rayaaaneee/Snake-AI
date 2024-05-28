import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Utile pour la sauvegarde du modèle
import os


class LinearQNet(nn.Module):
    def __init__(self, inputSize, hiddenSize, outputSize):
        super().__init__()
        self.linear1 = nn.Linear(inputSize, hiddenSize)
        self.linear2 = nn.Linear(hiddenSize, outputSize)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, filename='model.pth'):
        modelFolder = './model'
        if not os.path.exists(modelFolder):
            os.makedirs(modelFolder)
        filename = os.path.join(modelFolder, filename)
        torch.save(self.state_dict(), filename)


class QTrainer:

    def __init__(self, model, learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def trainStep(self, state, action, reward, nextState, done):
        """  # On convertit les numpy arrays en tensors
         state = torch.tensor(state, dtype=torch.float)
         nextState = torch.tensor(nextState, dtype=torch.float)
         action = torch.tensor(action, dtype=torch.long)
         reward = torch.tensor(reward, dtype=torch.float)

         # On met à jour le modèle
         if len(state.shape) == 1:
             # On ajoute une dimension si besoin
             # (car on a besoin d'un batch de données)
             # Exemple: [1, 2, 3] => [[1, 2, 3]]
             state = torch.unsqueeze(state, 0)
             nextState = torch.unsqueeze(nextState, 0)
             action = torch.unsqueeze(action, 0)
             reward = torch.unsqueeze(reward, 0)
             done = (done, )

         # 1: On calcule la prédiction Q(s, a)
         pred = self.model(state)

         # 2: On calcule la cible Q(s', a)
         target = pred.clone()
         for idx in range(len(done)):
             QNew = reward[idx]
             if not done[idx]:
                 QNew = reward[idx] + self.gamma * \
                     torch.max(self.model(nextState[idx]))

             target[idx][torch.argmax(action[idx]).item()] = QNew

         # 3: On calcule la loss
         self.optimizer.zero_grad()
         loss = self.criterion(target, pred)
         loss.backward()

         # 4: On fait une étape d'optimisation
         self.optimizer.step() """
