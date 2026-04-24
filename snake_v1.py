import math
import random
import sys

import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Grid Constants
WIDTH = 600
HEIGHT = 600
GRID_SIZE = 15
CELL_COUNT = WIDTH // GRID_SIZE  # 40 cells

# Colors
GREY = (30, 30, 30)
DARK_GREY = (20, 20, 20)
RED = (255, 60, 60)
GREEN = (0, 220, 80)
BLUE = (10, 127, 255)
BLACK = (0, 0, 0)
OFF_WHITE = (200, 200, 200)
YELLOW = (255, 215, 0)
PANEL_COLOR = (15, 15, 25)
ACCENT = (0, 180, 255)

# Directions (dx, dy)
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRS = [UP, RIGHT, DOWN, LEFT]  # clockwise order for turning

# GA hyperparameters
POPULATION_SIZE = 500
ELITE_FRACTION = 0.1
MUTATION_RATE = 0.05
MUTATION_STRENGTH = 0.3
MAX_STEPS_FACTOR = 100  # max steps = snake.length * MAX_STEPS_FACTOR (anti-loop)
MAX_STEPS_BASE = 200  # minimum steps even if length is 1


# ──────────────────────────────────────────────
#  Neural Network
# ──────────────────────────────────────────────
class NeuralNet:
    """
    24 inputs → 16 hidden (ReLU) → 16 hidden (ReLU) → 3 outputs (softmax)
    Outputs: [turn_left, go_straight, turn_right]
    """
    LAYER_SIZES = [24, 16, 16, 3]

    def __init__(self, weights=None):
        if weights is None:
            self.weights = self._random_weights()
        else:
            self.weights = weights

    def _random_weights(self):
        ws = []
        sizes = self.LAYER_SIZES
        for i in range(len(sizes) - 1):
            w = np.random.randn(sizes[i], sizes[i + 1]) * 0.5
            b = np.zeros(sizes[i + 1])
            ws.append((w, b))
        return ws

    def forward(self, x):
        for idx, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if idx < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
        # softmax
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def get_flat_weights(self):
        return np.concatenate([np.concatenate([w.flatten(), b]) for w, b in self.weights])

    @classmethod
    def from_flat_weights(cls, flat):
        sizes = cls.LAYER_SIZES
        net = cls.__new__(cls)
        net.weights = []
        idx = 0
        for i in range(len(sizes) - 1):
            n_w = sizes[i] * sizes[i + 1]
            n_b = sizes[i + 1]
            w = flat[idx:idx + n_w].reshape(sizes[i], sizes[i + 1])
            idx += n_w
            b = flat[idx:idx + n_b]
            idx += n_b
            net.weights.append((w, b))
        return net

    def crossover(self, other):
        a = self.get_flat_weights()
        b = other.get_flat_weights()
        mask = np.random.rand(len(a)) > 0.5
        child = np.where(mask, a, b)
        return NeuralNet.from_flat_weights(child)

    def mutate(self):
        flat = self.get_flat_weights()
        mask = np.random.rand(len(flat)) < MUTATION_RATE
        flat[mask] += np.random.randn(int(np.sum(mask))) * MUTATION_STRENGTH
        return NeuralNet.from_flat_weights(flat)


# ──────────────────────────────────────────────
#  Snake (headless, for training)
# ──────────────────────────────────────────────
class SnakeAgent:
    def __init__(self, brain: NeuralNet):
        self.brain = brain
        self.length = 2
        self.positions = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = random.choice(DIRS)
        self.alive = True
        self.steps = 0
        self.steps_since_eat = 0
        self.fruit = self._new_fruit()
        self.fitness = 0.0

    def _new_fruit(self):
        while True:
            x = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            y = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            if (x, y) not in self.positions:
                return x, y

    def get_inputs(self):
        """
        24 inputs: for each of 8 directions, 3 values:
          - normalised distance to wall   (1/dist)
          - 1 if body in that direction else 0
          - 1 if fruit in that direction else 0
        """
        head = self.positions[0]
        hx, hy = head
        fx, fy = self.fruit
        body_set = set(self.positions[1:])
        inputs = []

        eight_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]

        for dx, dy in eight_dirs:
            # wall distance
            dist = 1
            cx, cy = hx + dx * GRID_SIZE, hy + dy * GRID_SIZE
            body_found = 0
            fruit_found = 0
            while 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                if (cx, cy) in body_set and body_found == 0:
                    body_found = 1
                if (cx, cy) == (fx, fy):
                    fruit_found = 1
                cx += dx * GRID_SIZE
                cy += dy * GRID_SIZE
                dist += 1
            inputs.append(1.0 / dist)
            inputs.append(float(body_found))
            inputs.append(float(fruit_found))

        return np.array(inputs, dtype=np.float32)

    def step(self):
        if not self.alive:
            return

        # Get action from brain
        inputs = self.get_inputs()
        output = self.brain.forward(inputs)
        action = int(np.argmax(output))  # 0=left, 1=straight, 2=right

        # Convert relative action to absolute direction
        cur_idx = DIRS.index(self.direction)
        new_idx = (cur_idx + action - 1) % 4  # -1=left, 0=straight, +1=right
        self.direction = DIRS[new_idx]

        # Move
        hx, hy = self.positions[0]
        dx, dy = self.direction
        new_head = (
            (hx + dx * GRID_SIZE) % WIDTH,
            (hy + dy * GRID_SIZE) % HEIGHT,
        )

        # Collision with self
        if len(self.positions) > 2 and new_head in self.positions[1:]:
            self.alive = False
            return

        self.positions.insert(0, new_head)
        self.steps += 1
        self.steps_since_eat += 1

        # Eat fruit
        if new_head == self.fruit:
            self.length += 1
            self.steps_since_eat = 0
            self.fruit = self._new_fruit()
        else:
            if len(self.positions) > self.length:
                self.positions.pop()

        # Anti-loop: die if not eating
        max_steps = max(MAX_STEPS_BASE, self.length * MAX_STEPS_FACTOR)
        if self.steps_since_eat > max_steps:
            self.alive = False

    def compute_fitness(self):
        # Reward eating heavily, small bonus for surviving longer
        self.fitness = (self.length ** 2) * 100 + self.steps * 0.1
        return self.fitness

    def run_to_death(self):
        while self.alive:
            self.step()
        return self.compute_fitness()


# ──────────────────────────────────────────────
#  Genetic Algorithm
# ──────────────────────────────────────────────
class GeneticAlgorithm:
    def __init__(self):
        self.generation = 0
        self.population = [NeuralNet() for _ in range(POPULATION_SIZE)]
        self.best_brain = None
        self.best_fitness = 0
        self.history = []  # (gen, best, avg, max_length)

    def evolve(self):
        self.generation += 1

        # Evaluate all agents
        agents = [SnakeAgent(brain) for brain in self.population]
        fitnesses = [a.run_to_death() for a in agents]
        lengths = [a.length for a in agents]

        # Stats
        best_idx = int(np.argmax(fitnesses))
        best_fit = fitnesses[best_idx]
        avg_fit = float(np.mean(fitnesses))
        max_len = max(lengths)

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_brain = self.population[best_idx]

        self.history.append((self.generation, best_fit, avg_fit, max_len))

        # Selection: rank-based
        ranked = sorted(zip(fitnesses, self.population), key=lambda x: x[0], reverse=True)
        elite_n = max(2, int(POPULATION_SIZE * ELITE_FRACTION))
        elites = [brain for _, brain in ranked[:elite_n]]

        # Roulette wheel weights from fitness
        fit_arr = np.array(fitnesses, dtype=np.float64)
        fit_arr -= fit_arr.min()
        if fit_arr.sum() == 0:
            fit_arr = np.ones(len(fit_arr))
        fit_arr /= fit_arr.sum()

        # Build new population
        new_pop = list(elites)
        pop_list = self.population
        while len(new_pop) < POPULATION_SIZE:
            p1 = pop_list[np.random.choice(len(pop_list), p=fit_arr)]
            p2 = pop_list[np.random.choice(len(pop_list), p=fit_arr)]
            child = p1.crossover(p2).mutate()
            new_pop.append(child)

        self.population = new_pop
        return self.generation, best_fit, avg_fit, max_len, self.best_brain


# ──────────────────────────────────────────────
#  Playback: watch the best snake live
# ──────────────────────────────────────────────
class LiveSnake:
    """Visual snake driven by a NeuralNet brain."""

    def __init__(self, brain: NeuralNet):
        self.brain = brain
        self.length = 2
        self.positions = [(WIDTH // 2, HEIGHT // 2)]
        self.direction = random.choice(DIRS)
        self.alive = True
        self.steps_since_eat = 0
        self.fruit = self._new_fruit()

    def _new_fruit(self):
        while True:
            x = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            y = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            if (x, y) not in self.positions:
                return x, y

    def get_inputs(self):
        head = self.positions[0]
        hx, hy = head
        fx, fy = self.fruit
        body_set = set(self.positions[1:])
        inputs = []
        eight_dirs = [
            (0, -1), (1, -1), (1, 0), (1, 1),
            (0, 1), (-1, 1), (-1, 0), (-1, -1)
        ]
        for dx, dy in eight_dirs:
            dist = 1
            cx, cy = hx + dx * GRID_SIZE, hy + dy * GRID_SIZE
            body_found = 0
            fruit_found = 0
            while 0 <= cx < WIDTH and 0 <= cy < HEIGHT:
                if (cx, cy) in body_set and body_found == 0:
                    body_found = 1
                if (cx, cy) == (fx, fy):
                    fruit_found = 1
                cx += dx * GRID_SIZE
                cy += dy * GRID_SIZE
                dist += 1
            inputs.append(1.0 / dist)
            inputs.append(float(body_found))
            inputs.append(float(fruit_found))
        return np.array(inputs, dtype=np.float32)

    def update(self):
        if not self.alive:
            return
        inputs = self.get_inputs()
        output = self.brain.forward(inputs)
        action = int(np.argmax(output))
        cur_idx = DIRS.index(self.direction)
        new_idx = (cur_idx + action - 1) % 4
        self.direction = DIRS[new_idx]

        hx, hy = self.positions[0]
        dx, dy = self.direction
        new_head = (
            (hx + dx * GRID_SIZE) % WIDTH,
            (hy + dy * GRID_SIZE) % HEIGHT,
        )
        if len(self.positions) > 2 and new_head in self.positions[1:]:
            self.alive = False
            return

        self.positions.insert(0, new_head)
        self.steps_since_eat += 1

        if new_head == self.fruit:
            self.length += 1
            self.steps_since_eat = 0
            self.fruit = self._new_fruit()
        else:
            if len(self.positions) > self.length:
                self.positions.pop()

        if self.steps_since_eat > max(MAX_STEPS_BASE, self.length * MAX_STEPS_FACTOR):
            self.alive = False

    def render(self, surface):
        # Draw fruit
        fx, fy = self.fruit
        pygame.draw.circle(surface, RED, (fx, fy), GRID_SIZE // 2)

        # Draw body
        for idx, (x, y) in enumerate(self.positions):
            if idx == 0:
                pygame.draw.circle(surface, GREEN, (x, y), int(GRID_SIZE / 1.6))
                # Eyes
                if self.direction in (UP, DOWN):
                    pygame.draw.circle(surface, BLACK, (x + 4, y), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (x - 4, y), GRID_SIZE // 4)
                else:
                    pygame.draw.circle(surface, BLACK, (x, y + 4), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (x, y - 4), GRID_SIZE // 4)
            else:
                t = idx / max(len(self.positions), 1)
                r = int(10 + (1 - t) * (BLUE[0] - 10))
                g = int(100 + (1 - t) * (BLUE[1] - 100))
                b = int(200 + (1 - t) * (BLUE[2] - 200))
                pygame.draw.circle(surface, (r, g, b), (x, y), GRID_SIZE // 2)


# ──────────────────────────────────────────────
#  HUD / Stats Panel
# ──────────────────────────────────────────────
PANEL_W = 220


def draw_panel(screen, ga, gen, best_fit, avg_fit, max_len, snake):
    px = WIDTH + 8
    panel = pygame.Rect(px, 0, PANEL_W, HEIGHT + 8)
    pygame.draw.rect(screen, PANEL_COLOR, panel)
    pygame.draw.line(screen, ACCENT, (px, 0), (px, HEIGHT + 8), 1)

    font_title = pygame.font.SysFont("consolas", 14, bold=True)
    font_body = pygame.font.SysFont("consolas", 13)
    font_small = pygame.font.SysFont("consolas", 11)

    y = 16
    title = font_title.render("SNAKE  GA", True, ACCENT)
    screen.blit(title, (px + 12, y))
    y += 24
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    def row(label, val, color=OFF_WHITE):
        nonlocal y
        lbl = font_body.render(label, True, (120, 120, 140))
        v = font_body.render(str(val), True, color)
        screen.blit(lbl, (px + 12, y))
        screen.blit(v, (px + PANEL_W - v.get_width() - 12, y))
        y += 20

    row("Generation", gen, YELLOW)
    row("Score", snake.length, GREEN)
    row("Best Fit", f"{best_fit:.0f}", ACCENT)
    row("Avg Fit", f"{avg_fit:.0f}")
    row("Best Length", max_len, GREEN)
    row("Population", POPULATION_SIZE)
    row("Mutation", f"{MUTATION_RATE * 100:.0f}%")

    y += 8
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    # Mini fitness graph (last 20 gens)
    graph_label = font_small.render("FITNESS HISTORY", True, (100, 100, 120))
    screen.blit(graph_label, (px + 12, y))
    y += 14

    history = ga.history[-20:]
    if len(history) > 1:
        bests = [h[1] for h in history]
        avgs = [h[2] for h in history]
        max_v = max(bests) if max(bests) > 0 else 1
        gw, gh = PANEL_W - 24, 60
        gx, gy = px + 12, y
        pygame.draw.rect(screen, (25, 25, 40), (gx, gy, gw, gh))

        def plot_line(vals, color):
            pts = []
            for i, v in enumerate(vals):
                sx = gx + int(i / (len(vals) - 1) * gw)
                sy = gy + gh - int((v / max_v) * gh)
                pts.append((sx, sy))
            if len(pts) > 1:
                pygame.draw.lines(screen, color, False, pts, 1)

        plot_line(avgs, (60, 80, 120))
        plot_line(bests, ACCENT)
        y += gh + 8

    # Status
    status_color = GREEN if snake.alive else RED
    status_text = "RUNNING" if snake.alive else "DIED"
    st = font_body.render(status_text, True, status_color)
    screen.blit(st, (px + 12, y))
    y += 24

    hint = font_small.render("Q = quit  SPACE = skip", True, (70, 70, 90))
    screen.blit(hint, (px + 12, HEIGHT - 20))


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────
def main():
    total_w = WIDTH + 8 + PANEL_W
    screen = pygame.display.set_mode((total_w, HEIGHT + 8))
    pygame.display.set_caption("Snake — Neural Net Genetic Algorithm")
    surface = pygame.Surface((WIDTH + 8, HEIGHT + 8))
    surface = surface.convert()
    clock = pygame.time.Clock()

    ga = GeneticAlgorithm()

    print("Starting Genetic Algorithm training...")
    print(f"Population: {POPULATION_SIZE} | Network: 24→16→16→3")
    print("─" * 50)

    running = True
    while running:
        # ── Evolve one generation (blocking, but fast with numpy) ──
        print(f"Gen {ga.generation + 1}: evolving {POPULATION_SIZE} agents...", end=" ", flush=True)
        gen, best_fit, avg_fit, max_len, best_brain = ga.evolve()
        print(f"best={best_fit:.0f}  avg={avg_fit:.0f}  max_len={max_len}")

        # ── Replay the best brain visually ──
        live = LiveSnake(best_brain)
        skip_flag = False

        while live.alive and not skip_flag:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit()
                        sys.exit()
                    elif event.key == pygame.K_SPACE:
                        skip_flag = True

            live.update()

            # Draw game surface
            surface.fill(GREY)
            # Draw grid dots
            for gx in range(0, WIDTH, GRID_SIZE):
                for gy in range(0, HEIGHT, GRID_SIZE):
                    pygame.draw.circle(surface, (40, 40, 40), (gx + GRID_SIZE // 2, gy + GRID_SIZE // 2), 1)

            live.render(surface)

            screen.fill(DARK_GREY)
            screen.blit(surface, (0, 0))
            draw_panel(screen, ga, gen, best_fit, avg_fit, max_len, live)

            pygame.display.flip()
            level = math.ceil(live.length / 5)
            clock.tick(10 + level * 2)

        # Brief pause between generations
        if not skip_flag:
            pygame.time.wait(400)

        # Check quit during pause
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit()
                sys.exit()


if __name__ == "__main__":
    main()
