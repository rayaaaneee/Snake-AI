from classes.world import World
from classes.enums.direction import Direction
from classes.enums.color import Color

import pygame
import numpy as np
import random


class Interface():

    cellSize = 20

    screenSize = World.worldSize * 20

    gameHeightStartPos = 60

    lineWidth = 2

    font = "font/sometype.ttf"

    title = "SnakeGPT"

    # blocs / s
    speed = 15

    def __init__(self, isAI=False, isTraining=False):
        self.screen = pygame.display.set_mode(
            (Interface.screenSize,
             Interface.screenSize + Interface.gameHeightStartPos))
        pygame.display.set_caption(Interface.title)
        pygame.init()

        self._programIcon = pygame.image.load(
            "image/app_icon.png").convert_alpha()
        pygame.display.set_icon(self._programIcon)
        self._crownImage = pygame.image.load("image/crown.png")
        self._crownImage = pygame.transform.scale(
            self._crownImage, (30, 24)).convert_alpha()
        self._fontScore = pygame.font.Font(Interface.font, 20)
        self._fontMaxScore = pygame.font.Font(Interface.font, 17)

        self.cellSize = Interface.cellSize

        self.isAI = isAI

        self.isTraining = isTraining

        # Peut etre à supprimer étant donné ma condition
        self.frame = 0

        self._world = World(self.isAI, self.isTraining)

    def _updateDisplay(self) -> None:
        self.screen.fill(Color.BACKGROUND.value)
        pygame.draw.rect(self.screen, Color.TOP_BACKGROUND.value,
                         (0, 0, self.screenSize, self.gameHeightStartPos))
        pygame.draw.line(self.screen, Color.LINE.value,
                         (0, self.gameHeightStartPos - Interface.lineWidth),
                         (self.screenSize,
                          self.gameHeightStartPos - Interface.lineWidth),
                         Interface.lineWidth)
        self._setScore()
        self._setMaxScore()

    def update(self) -> None:
        if not self.isTraining:
            self._world.update()

        self._updateDisplay()
        self._drawApple()
        self._drawSnake()
        pygame.display.update()

    def _quit(self) -> None:
        pygame.quit()
        quit()

    def _setScore(self) -> None:
        score = self._world.score
        text = self._fontScore.render(
            "Score : " + str(score), True, (255, 255, 255))
        self.screen.blit(text, (
            (Interface.screenSize / 2) - (text.get_width() / 2),
            (Interface.gameHeightStartPos / 2) -
            (text.get_height() / 2) - Interface.lineWidth
        ))

    def _setMaxScore(self) -> None:
        maxScore = self._world.maxScore
        text = self._fontMaxScore.render(
            str(maxScore), True, (254, 183, 60))
        self.screen.blit(self._crownImage, (
            3,
            3))
        self.screen.blit(text, (self._crownImage.get_width() + 10, 8))

    def start(self) -> None:
        while True:
            clock = pygame.time.Clock()
            self.frame += 1
            self._world.snake.canChangeDirection = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._quit()
                # Si le jeu est lancé en mode humain, on récupère les touches
                if not self.isAI:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT and self._world.snake.direction != Direction.LEFT.value:
                            self._world.snake.changeDirection(
                                Direction.RIGHT.value)
                        elif event.key == pygame.K_LEFT and self._world.snake.direction != Direction.RIGHT.value:
                            self._world.snake.changeDirection(
                                Direction.LEFT.value)
                        elif event.key == pygame.K_UP and self._world.snake.direction != Direction.DOWN.value:
                            self._world.snake.changeDirection(
                                Direction.UP.value)
                        elif event.key == pygame.K_DOWN and self._world.snake.direction != Direction.UP.value:
                            self._world.snake.changeDirection(
                                Direction.DOWN.value)

            # Actions lors du jeu en mode IA
            if self.isAI:
                if self.isTraining:
                    self._world.trainer.train()

            self.update()
            clock.tick(Interface.speed)

    def _drawApple(self) -> None:

        pygame.draw.rect(
            self.screen,
            Color.APPLE.value,
            ((self._world.apple.position[0] * self.cellSize),
             (self._world.apple.position[1] * self.cellSize) +
             self.gameHeightStartPos,
             self.cellSize, self.cellSize)
        )

    def _drawSnake(self) -> None:

        for position in self._world.snake.positions:

            color = None

            if position == self._world.snake.getHead():
                color = Color.SNAKE_HEAD.value
            else:
                color = Color.SNAKE_BODY.value
            pygame.draw.rect(
                self.screen,
                color,
                (
                    (position[0] * self.cellSize),
                    (position[1] * self.cellSize) +
                    self.gameHeightStartPos,
                    self.cellSize,
                    self.cellSize
                )
            )
