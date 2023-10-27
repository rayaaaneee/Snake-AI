from classes.enums.direction import Direction
from classes.enums.reward import Reward


class Snake:

    color = 1

    initialLength = 3

    def __init__(self, world):
        self.world = world
        self.init()

    def init(self) -> None:
        self.length = Snake.initialLength
        self.direction = Direction.RIGHT.value
        self.positions = []
        for i in range(Snake.initialLength):
            self.positions.insert(0, (i, 0))

    def advance(self) -> None:
        newHead = self.getPosition(self.direction)
        self.positions.insert(0, newHead)

        # Si le serpent mange la pomme
        if len(self.positions) > self.length:
            self.positions.pop()

    def getPosition(self, direction) -> tuple[int, int]:
        head = self.getHead()
        if direction == Direction.RIGHT.value:
            return (head[0] + 1, head[1])
        elif direction == Direction.LEFT.value:
            return (head[0] - 1, head[1])
        elif direction == Direction.UP.value:
            return (head[0], head[1] - 1)
        elif direction == Direction.DOWN.value:
            return (head[0], head[1] + 1)

    def getHead(self) -> list[int]:
        return self.positions[0]

    def changeDirection(self, direction) -> None:
        if self.canChangeDirection:
            self._setDirection(direction)

    def _setDirection(self, direction) -> None:
        self.canChangeDirection = False
        self.direction = direction

    def isEating(self) -> bool:
        return self.getHead() == self.world.apple.position

    def _eatApple(self) -> None:
        self.length += 1
        self.world.apple.setNewPosition()
