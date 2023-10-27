import random


class Apple:

    color = 0

    def __init__(self, snake, world):
        self._world = world
        self._snake = snake
        self.setNewPosition()

    def setNewPosition(self) -> None:
        while True:
            x = random.randint(0, self._world.size - 1)
            y = random.randint(0, self._world.size - 1)
            if (x, y) not in self._snake.positions:
                self.position = (x, y)
                break
