import torch
import torch.nn as nn
import torch.optim as optim


class GlobalTrainer:

    def __init__(self, world, agent):
        self.plotScores = []
        self.avgScores = []
        self.totalScore = 0
        self.record = 0

        self._world = world
        self._agent = agent

    def train(self) -> None:
        oldState = self._agent.getState(self._world)

        # Nouvelle direction
        finalMove = self._agent.getAction(oldState)
        self._world.snake.changeDirection(finalMove)

        # variables
        reward, gameOver, score = self._world.update()

        newState = self._agent.getState(self._world)

        # Entrainer la mémoire à court terme (short memory)
        self._agent.trainShortMemory(
            oldState, finalMove, reward, newState, gameOver)

        # Remember
        self._agent.remember(oldState, finalMove, reward, newState, gameOver)

        if gameOver:
            self._agent.nbGames += 1
            self._agent.trainLongMemory()

            if (score > self.record):
                self.record = score
                self._agent.model.save()

            print("Game : " + str(self.nbGames) + " and Score : " +
                  str(score) + " and Record : " + str(self.record))
