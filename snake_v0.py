import math
import random
import sys
from typing import Optional

import numpy as np
import pygame

pygame.init()

# ── Grid Constants ─────────────────────────────────────────────────────────────
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 15
CELL_COUNT = WIDTH // GRID_SIZE  # 40 cells

# ── Colours ────────────────────────────────────────────────────────────────────
GREY = (30, 30, 30)
DARK_GREY = (20, 20, 20)
RED = (255, 60, 60)
GREEN = (0, 220, 80)
BLUE = (10, 127, 255)
BLACK = (0, 0, 0)
OFF_WHITE = (200, 200, 200)

# ── Directions (dx, dy) ────────────────────────────────────────────────────────
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRS = [UP, RIGHT, DOWN, LEFT]  # clockwise order for relative turns

# ── Game speed ─────────────────────────────────────────────────────────────────
FPS = 10  # steps per second — raise for AI training playback

# ── Anti-loop constants ────────────────────────────────────────────────────────
MAX_STEPS_BASE = 200
MAX_STEPS_FACTOR = 100


# ── Neural Network ─────────────────────────────────────────────────────────────
class NeuralNet:
    """
    24 inputs -> 16 hidden (ReLU) -> 16 hidden (ReLU) -> 3 outputs (softmax)
    Outputs: [turn_left, go_straight, turn_right]
    """
    LAYER_SIZES = [24, 16, 16, 3]

    def __init__(self, weights=None):
        self.weights = weights if weights is not None else self._random_weights()

    def _random_weights(self):
        ws = []
        for i in range(len(self.LAYER_SIZES) - 1):
            w = np.random.randn(self.LAYER_SIZES[i], self.LAYER_SIZES[i + 1]) * 0.5
            b = np.zeros(self.LAYER_SIZES[i + 1])
            ws.append((w, b))
        return ws

    def forward(self, x):
        for idx, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if idx < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
        e = np.exp(x - np.max(x))
        return e / np.sum(e)  # softmax

    def get_flat_weights(self):
        return np.concatenate([np.concatenate([w.flatten(), b])
                               for w, b in self.weights])

    @classmethod
    def from_flat_weights(cls, flat):
        net = cls.__new__(cls)
        net.weights = []
        idx = 0
        for i in range(len(cls.LAYER_SIZES) - 1):
            nw = cls.LAYER_SIZES[i] * cls.LAYER_SIZES[i + 1]
            nb = cls.LAYER_SIZES[i + 1]
            w = flat[idx:idx + nw].reshape(cls.LAYER_SIZES[i], cls.LAYER_SIZES[i + 1])
            idx += nw
            b = flat[idx:idx + nb]
            idx += nb
            net.weights.append((w, b))
        return net

    def crossover(self, other):
        a, b = self.get_flat_weights(), other.get_flat_weights()
        mask = np.random.rand(len(a)) > 0.5
        return NeuralNet.from_flat_weights(np.where(mask, a, b))

    def mutate(self, rate=0.05, strength=0.3):
        flat = self.get_flat_weights()
        mask = np.random.rand(len(flat)) < rate
        flat[mask] += np.random.randn(int(np.sum(mask))) * strength
        return NeuralNet.from_flat_weights(flat)


# ── Snake ──────────────────────────────────────────────────────────────────────
class Snake:
    """
    Moves every tick in self.direction.

    To steer:
      - Keyboard: call snake.set_direction("UP"/"DOWN"/"LEFT"/"RIGHT")
      - AI:       call snake.set_direction(0/1/2)  (relative: left/straight/right)
    """

    def __init__(self):
        cx = (CELL_COUNT // 2) * GRID_SIZE + GRID_SIZE // 2
        cy = (CELL_COUNT // 2) * GRID_SIZE + GRID_SIZE // 2
        self.positions = [(cx, cy)]
        self.direction = random.choice(DIRS)
        self.length = 2
        self.alive = True
        self.score = 0
        self.steps = 0
        self.steps_since_eat = 0
        self.fruit = self._new_fruit()

    def _new_fruit(self):
        while True:
            x = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            y = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            if (x, y) not in self.positions:
                return x, y

    def set_direction(self, action):
        """
        Update direction. Safe to call on every keypress or every AI frame.
          action = "UP"/"DOWN"/"LEFT"/"RIGHT"  -> absolute (keyboard)
          action = 0/1/2                       -> relative turn (AI)
        """
        abs_map = {"UP": UP, "DOWN": DOWN, "LEFT": LEFT, "RIGHT": RIGHT}
        if action in abs_map:
            new_dir = abs_map[action]
            cur_dx, cur_dy = self.direction
            # ignore 180 reversals
            if not (new_dir[0] == -cur_dx and new_dir[1] == -cur_dy):
                self.direction = new_dir
        elif action in (0, 1, 2):
            cur_idx = DIRS.index(self.direction)
            self.direction = DIRS[(cur_idx + int(action) - 1) % 4]

    def get_inputs(self):
        """
        Return 24-element float32 array: 8 directions x (dist, body, fruit).
        Raycasts wrap around the grid to match snake movement — no walls exist.
        dist = 1/steps so closer = higher value. Loop stops when ray returns to head.
        """
        hx, hy = self.positions[0]
        fx, fy = self.fruit
        body_set = set(self.positions[1:])
        eight_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
        ]
        inputs = []
        for dx, dy in eight_dirs:
            dist = 1
            cx = (hx + dx * GRID_SIZE) % WIDTH
            cy = (hy + dy * GRID_SIZE) % HEIGHT
            body_found = 0
            fruit_found = 0
            # travel until we wrap all the way back to the head
            while (cx, cy) != (hx, hy):
                if (cx, cy) in body_set and not body_found:
                    body_found = 1
                if (cx, cy) == (fx, fy):
                    fruit_found = 1
                cx = (cx + dx * GRID_SIZE) % WIDTH
                cy = (cy + dy * GRID_SIZE) % HEIGHT
                dist += 1
            inputs += [1.0 / dist, float(body_found), float(fruit_found)]
        return np.array(inputs, dtype=np.float32)

    def step(self):
        """Advance one cell in self.direction. Always moves — call once per tick."""
        if not self.alive:
            return

        hx, hy = self.positions[0]
        dx, dy = self.direction
        new_head = (
            (hx + dx * GRID_SIZE) % WIDTH,
            (hy + dy * GRID_SIZE) % HEIGHT,
        )

        # Self-collision
        if len(self.positions) > 2 and new_head in self.positions[1:]:
            self.alive = False
            return

        self.positions.insert(0, new_head)
        self.steps += 1
        self.steps_since_eat += 1

        if new_head == self.fruit:
            self.length += 1
            self.score += 1
            self.steps_since_eat = 0
            self.fruit = self._new_fruit()
        else:
            if len(self.positions) > self.length:
                self.positions.pop()

        # Anti-loop: die if not eating within time limit
        if self.steps_since_eat > max(MAX_STEPS_BASE, self.length * MAX_STEPS_FACTOR):
            self.alive = False


# ── Renderer ───────────────────────────────────────────────────────────────────
def render(surface, screen, snake):
    surface.fill(GREY)

    # Subtle dot grid
    for gx in range(0, WIDTH, GRID_SIZE):
        for gy in range(0, HEIGHT, GRID_SIZE):
            pygame.draw.circle(surface, (40, 40, 40),
                               (gx + GRID_SIZE // 2, gy + GRID_SIZE // 2), 1)

    # Fruit
    fx, fy = snake.fruit
    pygame.draw.circle(surface, RED, (fx, fy), GRID_SIZE // 2)

    # Snake
    for idx, (x, y) in enumerate(snake.positions):
        if idx == 0:
            pygame.draw.circle(surface, GREEN, (x, y), int(GRID_SIZE / 1.6))
            if snake.direction in (UP, DOWN):
                pygame.draw.circle(surface, BLACK, (x + 4, y), GRID_SIZE // 4)
                pygame.draw.circle(surface, BLACK, (x - 4, y), GRID_SIZE // 4)
            else:
                pygame.draw.circle(surface, BLACK, (x, y + 4), GRID_SIZE // 4)
                pygame.draw.circle(surface, BLACK, (x, y - 4), GRID_SIZE // 4)
        else:
            t = idx / max(len(snake.positions), 1)
            col = (
                int(10 + (1 - t) * (BLUE[0] - 10)),
                int(100 + (1 - t) * (BLUE[1] - 100)),
                int(200 + (1 - t) * (BLUE[2] - 200)),
            )
            pygame.draw.circle(surface, col, (x, y), GRID_SIZE // 2)

    # Minimal score
    font = pygame.font.SysFont("consolas", 18, bold=True)
    label = font.render(f"Score: {snake.score}", True, OFF_WHITE)
    surface.blit(label, (10, 8))

    screen.fill(DARK_GREY)
    screen.blit(surface, (0, 0))
    pygame.display.flip()


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Snake")
    surface = pygame.Surface((WIDTH, HEIGHT)).convert()
    clock = pygame.time.Clock()

    # To switch to AI: assign a NeuralNet instance here
    ai_brain = None  # type: Optional[NeuralNet]
    ai_brain = NeuralNet()  # uncomment for a (random) AI brain

    snake = Snake()

    while True:
        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                if ai_brain is None:
                    if event.key in (pygame.K_UP, pygame.K_w): snake.set_direction("UP")
                    if event.key in (pygame.K_DOWN, pygame.K_s): snake.set_direction("DOWN")
                    if event.key in (pygame.K_LEFT, pygame.K_a): snake.set_direction("LEFT")
                    if event.key in (pygame.K_RIGHT, pygame.K_d): snake.set_direction("RIGHT")

        # ── AI steering (every frame) ─────────────────────────────────────────
        if ai_brain is not None and snake.alive:
            output = ai_brain.forward(snake.get_inputs())
            snake.set_direction(int(np.argmax(output)))  # 0/1/2

        # ── Step & render ─────────────────────────────────────────────────────
        snake.step()
        render(surface, screen, snake)

        if not snake.alive:
            pygame.time.wait(1000)
            snake = Snake()

        level = math.ceil(snake.length / 5)
        clock.tick(10 + level * 2)


if __name__ == "__main__":
    main()
