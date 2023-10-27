from classes.snake import Snake
from classes.apple import Apple
from classes.enums.color import Color
from classes.enums.reward import Reward

from network.agent import Agent
from network.global_trainer import GlobalTrainer
import pygame
import json


class World:

    worldSize = 20

    def __init__(self, isAi, isTraining):

        self.isAI = isAi
        self.isTraining = isTraining

        jsonFile = "ai" if self.isAI else "human"
        self._jsonPath = "data/" + jsonFile + ".json"
        self._json = json.load(open(self._jsonPath, "r"))
        self.maxScore = self._json["maxScore"]

        self.score = 0

        self.size = World.worldSize

        self.snake = Snake(self)
        self.apple = Apple(self.snake, self)

        if self.isAI:

            self.reward = 0
            self.agent = Agent()

            if self.isTraining:
                self.trainer = GlobalTrainer(self, self.agent)

    def update(self) -> (int, bool, int):

        reward = Reward.MOVE.value

        self.snake.advance()

        isDead = self.isCollision()

        if self.snake.isEating():

            self.snake._eatApple()
            self.score = self.snake.length - Snake.initialLength

            if (self.score > self.maxScore):

                self.maxScore = self.score
                self._writeMaxScore()

            reward = Reward.EAT.value

        score = self.score

        if isDead:
            self._restart()
            reward = Reward.DIE.value

        return reward, isDead, score

    def isCollision(self, pt=None) -> bool:
        head = self.snake.getHead()
        if pt is not None:
            head = pt
        isDead = False
        if (head[0] < 0) or (head[0] > self.size - 1) or (head[1] < 0) or (head[1] > self.size - 1):
            isDead = True
        elif head in self.snake.positions[1:]:
            isDead = True

        return isDead

    def _writeMaxScore(self) -> None:
        self._json["maxScore"] = self.maxScore
        with open(self._jsonPath, "w") as outfile:
            json.dump(self._json, outfile, indent=4)

    def _restart(self) -> None:
        self.score = 0
        self.snake.init()
        self.apple.setNewPosition()

    def updateReward(self, reward) -> None:
        self.reward += reward
