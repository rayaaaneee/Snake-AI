from classes.direction import Direction
import json


class Snake:

    color = 1

    initialLength = 3

    def __init__(self, world):
        self.world = world
        isAI = self.world.interface.isAI
        json_file = "ai" if isAI else "human"
        self.json_path = "data/" + json_file + ".json"
        self.json = json.load(open(self.json_path, "r"))
        self.maxScore = self.json["maxScore"]
        self.init()

    def init(self):
        self.length = Snake.initialLength
        self.direction = Direction.RIGHT.value
        self.score = 0
        self.positions = []
        for i in range(Snake.initialLength):
            self.positions.insert(0, (i, 0))

    def advance(self):
        head = self.positions[0]
        if self.direction == Direction.RIGHT.value:
            newHead = (head[0] + 1, head[1])
        elif self.direction == Direction.LEFT.value:
            newHead = (head[0] - 1, head[1])
        elif self.direction == Direction.UP.value:
            newHead = (head[0], head[1] - 1)
        elif self.direction == Direction.DOWN.value:
            newHead = (head[0], head[1] + 1)
        self.positions.insert(0, newHead)

        # Si le serpent mange la pomme
        if len(self.positions) > self.length:
            self.positions.pop()

    def changeDirection(self, direction):
        if self.canChangeDirection:
            self.setDirection(direction)

    def setDirection(self, direction):
        self.canChangeDirection = False
        self.direction = direction

    def eatApple(self):
        self.length += 1
        self.score = self.length - Snake.initialLength

        if (self.score > self.maxScore):
            self.maxScore = self.score
            self.writeMaxScore()

        self.world.apple.setNewPosition()

    def writeMaxScore(self):
        self.json["maxScore"] = self.maxScore
        with open(self.json_path, "w") as outfile:
            json.dump(self.json, outfile, indent=4)
