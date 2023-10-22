import random


class Apple:

    color = 0

    def __init__(self, snake, world):
        self.world = world
        self.snake = snake
        self.setNewPosition()

    def setNewPosition(self):
        while True:
            x = random.randint(0, self.world.size - 1)
            y = random.randint(0, self.world.size - 1)
            if (x, y) not in self.snake.positions:
                self.position = (x, y)
                break
