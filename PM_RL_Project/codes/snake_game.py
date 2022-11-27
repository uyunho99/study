import numpy as np
import gym  # 강화학습 환경을 제공하는 라이브러리
import pygame  # 게임을 만들기 위한 라이브러리

from gym import spaces  # action space, observation space를 정의하기 위한 라이브러리

pygame.init()  # pygame 초기화


def to_rgb(color: str):
    return tuple(int(color[i:i+2], 16) for i in (1, 3, 5))


BLACK = to_rgb('#000000')
WHITE = to_rgb('#FFFFFF')
RED = to_rgb('#d4474d')
GREEN = to_rgb('#4dd447')
BLUE = to_rgb('#4474d4')

# === Screen ===
screen_size = (600, 600)  # 스크린 픽셀 수
grid_size = (20, 20)  # 스크린 grid_size 개수로 나눔

# 픽셀 위치


def get_pos(grid_pos, grid_size=grid_size):
    return (grid_pos[0] * (screen_size[0] // grid_size[0]), grid_pos[1] * (screen_size[1] // grid_size[1]))

# 그리드 위치


def get_grid_pos(pos, grid_size=grid_size):
    return (pos[0] // (screen_size[0] // grid_size[0]), pos[1] // (screen_size[1] // grid_size[1]))


# 스크린 생성
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption('Snake Game')

# === Fonts ===
font = pygame.font.SysFont('Poppins', size=20, bold=False, italic=False)


# === Draw ===
def draw_block(grid_pos, color, grid_size=grid_size):
    pos = get_pos(grid_pos, grid_size)
    block_size = (screen_size[0] // grid_size[0],
                  screen_size[1] // grid_size[1])
    pygame.draw.rect(
        screen, color, (pos[0], pos[1], block_size[0], block_size[1]))


def draw_text(text, grid_pos, color=WHITE):
    pos = get_pos(grid_pos)
    text = font.render(text, antialias=True, color=color, background=None)
    screen.blit(text, pos)


KEY_DIRECTION = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3
}


class Snake(gym.Env):
    def __init__(self):
        self.body = [(2, 2), (2, 3), (2, 4)]
        self.food = None
        self.board_size = np.array(grid_size)
        self.direction = 2  # 0: up, 1: right, 2: down, 3: left

        self.now = 0
        self.last_eat = 0

        # image observation
        # self.observation_space = spaces.Box(low=0, high=255, shape=(self.board_size[0], self.board_size[1], 3), dtype = np.uint8)

        # array observation
        self.observation_space = spaces.Box(low=0, high=2, shape=(
            self.board_size[0], self.board_size[1]), dtype=np.uint8)

        self.action_space = spaces.Discrete(4)

        self.reset()

    def get_obs(self):
        board = np.zeros(self.board_size, dtype=np.uint8)
        for pos in self.body:
            board[pos] = 1
        board[self.food] = 2

        return board

    def reset(self):
        self.direction = 1
        self.body = [(2, 2), (2, 3), (2, 4)]
        # self.body = deque([(2, 2), (2, 3), (2, 4)])
        self.food = self.generate_food()

        self.now = 0
        self.last_eat = 0

        return self.get_obs()

    def move(self):
        head = self.body[0]
        grown = False
        if self.direction == 0:
            head = (head[0], head[1] - 1)
        elif self.direction == 1:
            head = (head[0] + 1, head[1])
        elif self.direction == 2:
            head = (head[0], head[1] + 1)
        elif self.direction == 3:
            head = (head[0] - 1, head[1])
        # self.body.insert(0, head)
        self.body.appendleft(head)

        if head == self.food:
            self.food = self.generate_food()
            self.last_eat = self.now
            grown = True
        else:
            self.body.pop()

        return head, grown

    def step(self, action):
        done, info = False, {}
        reward = 0
        self.now += 1
        self.direction = action
        head, grown = self.move()

        if self.now - self.last_eat > np.sum(self.board_size) * 2:
            done = True
            info['msg'] = 'timeout'
            reward = -10

        elif head[0] < 0 or head[0] >= self.board_size[0] or head[1] < 0 or head[1] >= self.board_size[1]:
            done = True
            info['msg'] = 'wall'
            reward = -10

        elif head in list(self.body)[1:]:
            done = True
            info['msg'] = 'body'
            reward = -10

        if done and info['msg'] in ['wall', 'body']:
            # self.body.popleft()
            self.body.pop(0)

        if grown:
            reward = 1

        return self.get_obs(), reward, done, info

    def generate_food(self):
        while True:
            food = (np.random.randint(0, self.board_size[0]), np.random.randint(
                0, self.board_size[1]))
            if food not in self.body:
                return food

        #coords = list(zip(range(self.board_size[0]), range(self.board_size[1])))
        #coords = [c for c in coords if c not in self.body]
        # return random.choice(coords)

    def draw(self):
        for pos in self.body:
            draw_bllock(pos, GREEN)
        draw_block(self.food, RED)


def main():
    clock = pygame.time.Clock()
    snake = Snake()
    done = False
    pygame_done = False

    # res_reward = 0

    while not done:
        clock.tick(5)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key in KEY_DIRECTION:
                    snake.direction = KEY_DIRECTION[event.key]
                elif event.key == pygame.K_q:
                    pygame_done = True

        obs, reward, done, info = snake.step(snake.direction)
        res_reward += reward

        screen.fill(BLACK)
        snake.draw()
        pygame.display.flip()

    print(f'Reward: {res_reward}, Info: {info}')
    # wait for close the window
    while not pygame_done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame_done = True
    pygame.quit()


if __name__ == "__main__":
    main()
