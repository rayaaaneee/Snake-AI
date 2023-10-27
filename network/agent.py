import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque
import numpy as np
import random

from classes.enums.direction import Direction
from network.model import LinearQNet, QTrainer


class Agent():

    MAX_MEMORY = 100_000
    BATCH_SIZE = 1000
    LEARNING_RATE = 0.001  # Alpha

    def __init__(self):
        self.nbGames = 0
        self.epsilon = 0  # Taux de random
        self.gamma = 0.9  # Taux de réduction
        # Lorsque la mémoire est pleine, on supprime les anciennes expériences
        self.memory = deque(maxlen=Agent.MAX_MEMORY)

        # Il y a 11 paramètres en entrée et 3 paramètres en sortie, la valeur
        # 256 est arbitraire et dépend du besoin
        self.model = LinearQNet(11, 256, 3)
        self.trainer = QTrainer(self.model, Agent.LEARNING_RATE, self.gamma)

    @staticmethod
    def interpretDirection(self, action) -> int:
        orderClockWise = [Direction.RIGHT.value, Direction.DOWN.value,
                          Direction.LEFT.value, Direction.UP.value]
        index = orderClockWise.index(self._world.snake.direction)

        if np.array_equal(action, [1, 0, 0]):
            # On avance tout droit
            pass
        elif np.array_equal(action, [0, 1, 0]):
            # On tourne à droite (peu importe la direction)
            index = (index + 1) % 4
        elif np.array_equal(action, [0, 0, 1]):
            # On tourne à gauche (peu importe la direction)
            index = (index - 1) % 4
        else:
            # Tableau non conforme, on pourrait quitter sur une erreur
            # mais on fera plutot du random
            index = random.randint(0, (len(orderClockWise) - 1))
        return orderClockWise[index]

    def getState(self, world) -> []:

        pointLeft = world.snake.getPosition(Direction.LEFT.value)
        pointRight = world.snake.getPosition(Direction.RIGHT.value)
        pointUp = world.snake.getPosition(Direction.UP.value)
        pointDown = world.snake.getPosition(Direction.DOWN.value)

        isDirectionLeft = world.snake.direction == Direction.LEFT.value
        isDirectionRight = world.snake.direction == Direction.RIGHT.value
        isDirectionUp = world.snake.direction == Direction.UP.value
        isDirectionDown = world.snake.direction == Direction.DOWN.value

        state = [
            # Danger straight -> Danger droit
            (isDirectionLeft and world.isCollision(pointLeft)) or
            (isDirectionRight and world.isCollision(pointRight)) or
            (isDirectionUp and world.isCollision(pointUp)) or
            (isDirectionDown and world.isCollision(pointDown)),

            # Danger à droite
            (isDirectionLeft and world.isCollision(pointDown)) or
            (isDirectionRight and world.isCollision(pointUp)) or
            (isDirectionUp and world.isCollision(pointRight)) or
            (isDirectionDown and world.isCollision(pointLeft)),

            # Danger à gauche
            (isDirectionLeft and world.isCollision(pointUp)) or
            (isDirectionRight and world.isCollision(pointDown)) or
            (isDirectionUp and world.isCollision(pointLeft)) or
            (isDirectionDown and world.isCollision(pointRight)),

            # Direction
            isDirectionLeft,
            isDirectionRight,
            isDirectionUp,
            isDirectionDown,

            # Emplacement de la pomme
            world.apple.position[0] < world.snake.getHead()[
                0],  # Pomme à gauche
            world.apple.position[0] > world.snake.getHead()[
                0],  # Pomme à droite
            world.apple.position[1] < world.snake.getHead()[
                1],  # Pomme en haut
            world.apple.position[1] > world.snake.getHead()[1]   # Pomme en bas
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, currentGameOver):
        self.memory.append((state, action, reward, nextState, currentGameOver))

    def trainLongMemory(self):
        if len(self.memory) > self.BATCH_SIZE:
            miniSample = random.sample(self.memory, self.BATCH_SIZE)
        else:
            miniSample = self.memory

        states, actions, rewards, nextStates, currentGameOvers = zip(
            *miniSample)
        self.trainer.trainStep(states, actions, rewards,
                               nextStates, currentGameOvers)
        # for state, action, reward, nextState, currentGameOver in miniSample:
        #    self.trainer.trainStep(state, action, reward, nextState, currentGameOver)

    def trainShortMemory(self, state, action, reward, nextState, currentGameOver):
        self.trainer.trainStep(state, action, reward,
                               nextState, currentGameOver)

    def getAction(self, state) -> [int, int, int]:
        # Mouvements aléatoires : exploration/ Prédiction : exploitation
        # 80 et 200 sont des valeurs arbitraires
        self.epsilon = 80 - self.nbGames  # Plus on joue moins on explore

        finalMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            # On explore
            move = random.randint(0, len(finalMove) - 1)
            finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            # Appelera la fonction forward de LinearQNet
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove
