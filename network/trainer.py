import torch
import torch.nn as nn
import torch.optim as optim

class SnakeTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

    def train(self, input, target):
        self.optimizer.zero_grad()
        output = self.model(input)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        return loss.item()
