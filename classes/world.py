from classes.snake import Snake
from classes.apple import Apple
from classes.color import Color
import pygame


class World:

    def __init__(self, size, interface):
        self.size = size
        self.interface = interface
        self.snake = Snake(self)
        self.apple = Apple(self.snake, self)

    def update(self):
        self.snake.advance()

        self.checkCollision()

        self.drawApple()
        self.drawSnake()

        if self.snake.positions[0] == self.apple.position:
            self.snake.eatApple()

    def checkCollision(self):
        head = self.snake.positions[0]
        if (head[0] < 0) or (head[0] > self.size - 1) or (head[1] < 0) or (head[1] > self.size - 1):
            self.restart()
        elif head in self.snake.positions[1:]:
            self.restart()

    def restart(self):
        self.snake.init()
        self.apple.setNewPosition()

    def drawApple(self):
        pygame.draw.rect(
            self.interface.screen,
            Color.APPLE,
            ((self.apple.position[0] * self.interface.cellSize),
             (self.apple.position[1] * self.interface.cellSize) +
             self.interface.gameHeightStartPos,
             self.interface.cellSize, self.interface.cellSize)
        )

    def drawSnake(self):
        for position in self.snake.positions:
            pygame.draw.rect(
                self.interface.screen,
                Color.SNAKE,
                (
                    (position[0] * self.interface.cellSize),
                    (position[1] * self.interface.cellSize) +
                    self.interface.gameHeightStartPos,
                    self.interface.cellSize,
                    self.interface.cellSize
                )
            )
