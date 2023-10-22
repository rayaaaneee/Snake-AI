from classes.world import World
from classes.direction import Direction
from classes.color import Color
import pygame


class Interface():

    cellSize = 20

    worldSize = 20

    screenSize = worldSize * 20

    gameHeightStartPos = 60

    lineWidth = 2

    font = "font/sometype.ttf"

    title = "SnakeGPT"

    # blocs / s
    speed = 15

    def __init__(self, isAI=False):
        self.screen = pygame.display.set_mode(
            (Interface.screenSize,
             Interface.screenSize + Interface.gameHeightStartPos))
        pygame.display.set_caption(Interface.title)
        self.programIcon = pygame.image.load(
            "image/app_icon.png").convert_alpha()
        pygame.display.set_icon(self.programIcon)
        pygame.init()

        self.cellSize = Interface.cellSize
        self.fontScore = pygame.font.Font(Interface.font, 20)
        self.fontMaxScore = pygame.font.Font(Interface.font, 17)
        self.crownImage = pygame.image.load("image/crown.png")
        self.crownImage = pygame.transform.scale(
            self.crownImage, (30, 24)).convert_alpha()
        self.world = World(Interface.worldSize, self)
        self.isAI = isAI

    def updateDisplay(self):
        self.screen.fill(Color.BACKGROUND.value)
        pygame.draw.rect(self.screen, Color.TOP_BACKGROUND.value,
                         (0, 0, self.screenSize, self.gameHeightStartPos))
        pygame.draw.line(self.screen, Color.LINE.value,
                         (0, self.gameHeightStartPos - Interface.lineWidth),
                         (self.screenSize,
                          self.gameHeightStartPos - Interface.lineWidth),
                         Interface.lineWidth)
        self.world.update()
        self.setScore()
        self.setMaxScore()
        pygame.display.update()

    def setScore(self):
        score = self.world.snake.score
        text = self.fontScore.render(
            "Score : " + str(score), True, (255, 255, 255))
        self.screen.blit(text, (
            (Interface.screenSize / 2) - (text.get_width() / 2),
            (Interface.gameHeightStartPos / 2) -
            (text.get_height() / 2) - Interface.lineWidth
        ))

    def setMaxScore(self):
        maxScore = self.world.snake.maxScore
        text = self.fontMaxScore.render(
            str(maxScore), True, (254, 183, 60))
        self.screen.blit(self.crownImage, (
            3,
            3))
        self.screen.blit(text, (self.crownImage.get_width() + 10, 8))

    def start(self):
        while True:
            clock = pygame.time.Clock()
            self.world.snake.canChangeDirection = True
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if not self.isAI:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_RIGHT and self.world.snake.direction != Direction.LEFT.value:
                            self.world.snake.changeDirection(
                                Direction.RIGHT.value)
                        elif event.key == pygame.K_LEFT and self.world.snake.direction != Direction.RIGHT.value:
                            self.world.snake.changeDirection(
                                Direction.LEFT.value)
                        elif event.key == pygame.K_UP and self.world.snake.direction != Direction.DOWN.value:
                            self.world.snake.changeDirection(
                                Direction.UP.value)
                        elif event.key == pygame.K_DOWN and self.world.snake.direction != Direction.UP.value:
                            self.world.snake.changeDirection(
                                Direction.DOWN.value)
                else:
                    pass
            self.updateDisplay()
            clock.tick(Interface.speed)
