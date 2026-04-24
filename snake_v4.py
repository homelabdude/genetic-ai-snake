import math
import random
import sys
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pygame

pygame.init()

# ── Grid Constants ─────────────────────────────────────────────────────────────
WIDTH      = 600
HEIGHT     = 600
GRID_SIZE  = 15
CELL_COUNT = WIDTH // GRID_SIZE

# ── Colours ────────────────────────────────────────────────────────────────────
GREY        = (30,  30,  30)
DARK_GREY   = (20,  20,  20)
RED         = (255, 60,  60)
GREEN       = (0,   220, 80)
BLUE        = (10,  127, 255)
BLACK       = (0,   0,   0)
OFF_WHITE   = (200, 200, 200)
YELLOW      = (255, 215, 0)
PANEL_COLOR = (15,  15,  25)
ACCENT      = (0,   180, 255)

# ── Directions ─────────────────────────────────────────────────────────────────
UP    = ( 0, -1)
DOWN  = ( 0,  1)
LEFT  = (-1,  0)
RIGHT = ( 1,  0)
DIRS  = [UP, RIGHT, DOWN, LEFT]

# ── GA Hyperparameters ─────────────────────────────────────────────────────────
POPULATION_SIZE   = 1000
ELITE_FRACTION    = 0.08
MUTATION_RATE     = 0.03
MUTATION_STRENGTH = 0.2
TOURNAMENT_K      = 4     # competitors per tournament selection
RANDOM_INJECT     = 0.02  # fraction replaced with fresh random nets each gen

# ── Anti-circle params ─────────────────────────────────────────────────────────
MAX_STEPS_NO_EAT = 150    # tighter — lost snakes die cleanly
REVISIT_PENALTY  = 7.0   # high enough to hurt without needing loop-death backup
VISIT_DECAY      = 0.5    # multiply visit counts on eat, don't wipe

# ── HUD ────────────────────────────────────────────────────────────────────────
PANEL_W = 220


# ── Neural Network  29 → 24 → 16 → 3 ──────────────────────────────────────────
class NeuralNet:
    LAYER_SIZES = [29, 24, 16, 3]   # 29 = 24 raycast + 4 fruit direction + 1 reachability

    def __init__(self, weights=None):
        self.weights = weights if weights is not None else self._random_weights()

    def _random_weights(self):
        ws = []
        for i in range(len(self.LAYER_SIZES) - 1):
            fan_in = self.LAYER_SIZES[i]
            w = np.random.randn(fan_in, self.LAYER_SIZES[i + 1]) * math.sqrt(2.0 / fan_in)
            b = np.zeros(self.LAYER_SIZES[i + 1])
            ws.append((w, b))
        return ws

    def forward(self, x):
        for i, (w, b) in enumerate(self.weights):
            x = x @ w + b
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)   # ReLU; final layer returns raw logits
        return x

    def get_flat_weights(self):
        return np.concatenate([np.concatenate([w.ravel(), b])
                               for w, b in self.weights])

    @classmethod
    def from_flat_weights(cls, flat):
        net = cls.__new__(cls)
        net.weights = []
        idx = 0
        for i in range(len(cls.LAYER_SIZES) - 1):
            r, c = cls.LAYER_SIZES[i], cls.LAYER_SIZES[i + 1]
            w = flat[idx:idx + r * c].reshape(r, c); idx += r * c
            b = flat[idx:idx + c];                   idx += c
            net.weights.append((w, b))
        return net

    def crossover(self, other):
        a, b = self.get_flat_weights(), other.get_flat_weights()
        cut  = random.randint(1, len(a) - 1)
        return NeuralNet.from_flat_weights(np.concatenate([a[:cut], b[cut:]]))

    def mutate(self):
        flat = self.get_flat_weights().copy()
        mask = np.random.rand(len(flat)) < MUTATION_RATE
        flat[mask] += np.random.randn(int(np.sum(mask))) * MUTATION_STRENGTH
        return NeuralNet.from_flat_weights(flat)


# ── Snake Agent (headless training) ───────────────────────────────────────────
class SnakeAgent:
    def __init__(self, brain: NeuralNet):
        self.brain   = brain
        self.length  = 2
        # Random start — breaks the consistent first-fruit geometry
        # that locks in a turning bias when always starting dead centre
        cx = random.randint(2, CELL_COUNT - 3) * GRID_SIZE + GRID_SIZE // 2
        cy = random.randint(2, CELL_COUNT - 3) * GRID_SIZE + GRID_SIZE // 2
        self.positions      = [(cx, cy)]
        self.direction      = random.choice(DIRS)
        self.alive          = True
        self.steps          = 0
        self.steps_no_eat   = 0
        self.fruit          = self._new_fruit()
        self.fitness        = 0.0
        self.cell_visits    = {(cx, cy): 1}
        self.recent_heads   = []
        self.circle_penalty = 0.0

    def _new_fruit(self):
        pos_set = set(self.positions)
        while True:
            x = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            y = random.randint(0, CELL_COUNT - 1) * GRID_SIZE + GRID_SIZE // 2
            if (x, y) not in pos_set:
                return x, y

    def _reachable_cells(self, max_depth=20):
        """BFS from head — count reachable cells up to max_depth steps.
        Normalised to 0-1. Gives the snake a sense of whether it's about
        to trap itself, which raycasts can't provide at high lengths."""
        head     = self.positions[0]
        body_set = set(self.positions[1:])
        visited  = {head}
        queue    = [(head, 0)]
        count    = 0
        while queue:
            (cx, cy), depth = queue.pop(0)
            if depth >= max_depth:
                continue
            for ddx, ddy in DIRS:
                nx   = (cx + ddx * GRID_SIZE) % WIDTH
                ny   = (cy + ddy * GRID_SIZE) % HEIGHT
                npos = (nx, ny)
                if npos not in visited and npos not in body_set:
                    visited.add(npos)
                    count += 1
                    queue.append((npos, depth + 1))
        return min(1.0, count / (max_depth * max_depth))

    def get_inputs(self):
        """
        29-element float32 array:
          - 8 directions x (dist, body, fruit) = 24 raycast values
          - normalised dx/dy to fruit + binary quadrant flags  = 4 values
          - reachable cells BFS score                          = 1 value
        """
        hx, hy     = self.positions[0]
        fx, fy     = self.fruit
        body_set   = set(self.positions[1:])
        eight_dirs = [
            ( 0, -1), ( 1, -1), ( 1,  0), ( 1,  1),
            ( 0,  1), (-1,  1), (-1,  0), (-1, -1),
        ]
        inputs = []
        for dx, dy in eight_dirs:
            dist        = 1
            cx = (hx + dx * GRID_SIZE) % WIDTH
            cy = (hy + dy * GRID_SIZE) % HEIGHT
            body_found  = 0
            fruit_found = 0
            while dist <= CELL_COUNT:
                if (cx, cy) in body_set and body_found == 0:
                    body_found  = 1
                if (cx, cy) == (fx, fy):
                    fruit_found = 1
                cx = (cx + dx * GRID_SIZE) % WIDTH
                cy = (cy + dy * GRID_SIZE) % HEIGHT
                dist += 1
            inputs += [1.0 / dist, float(body_found), float(fruit_found)]

        # Fruit direction — shortest-path aware (wrapping torus)
        raw_dx = fx - hx
        raw_dy = fy - hy
        if abs(raw_dx) > WIDTH  // 2: raw_dx -= int(math.copysign(WIDTH,  raw_dx))
        if abs(raw_dy) > HEIGHT // 2: raw_dy -= int(math.copysign(HEIGHT, raw_dy))
        inputs += [
            raw_dx / WIDTH,
            raw_dy / HEIGHT,
            float(raw_dx > 0),
            float(raw_dy > 0),
            self._reachable_cells(),
        ]

        return np.array(inputs, dtype=np.float32)

    def step(self):
        if not self.alive:
            return

        output         = self.brain.forward(self.get_inputs())
        action         = int(np.argmax(output))
        cur_idx        = DIRS.index(self.direction)
        self.direction = DIRS[(cur_idx + action - 1) % 4]

        hx, hy   = self.positions[0]
        dx, dy   = self.direction
        new_head = (
            (hx + dx * GRID_SIZE) % WIDTH,
            (hy + dy * GRID_SIZE) % HEIGHT,
        )

        if len(self.positions) > 2 and new_head in set(self.positions[1:]):
            self.alive = False
            return

        self.positions.insert(0, new_head)
        self.steps        += 1
        self.steps_no_eat += 1

        # Revisit penalty — scales super-linearly so repeated loops hurt fast
        visits = self.cell_visits.get(new_head, 0)
        if visits > 0:
            self.circle_penalty += REVISIT_PENALTY * (visits ** 1.5)
        self.cell_visits[new_head] = visits + 1

        if new_head == self.fruit:
            self.length       += 1
            self.steps_no_eat  = 0
            # Decay rather than wipe — historical path still costs something
            self.cell_visits  = {k: max(1, int(v * VISIT_DECAY))
                                 for k, v in self.cell_visits.items()}
            self.recent_heads = []
            self.fruit        = self._new_fruit()
        else:
            if len(self.positions) > self.length:
                self.positions.pop()

        if self.steps_no_eat > MAX_STEPS_NO_EAT:
            self.alive = False

    def compute_fitness(self):
        score = (self.length ** 2) * 500
        # No step bonus — rewarding survival independent of eating
        # caused circling strategies to outcompete eating ones.
        # Penalty scales with length so looping stays costly at high scores.
        scaled_penalty = self.circle_penalty * max(1, self.length // 4)
        self.fitness   = max(0.0, score - scaled_penalty)
        return self.fitness

    def run_to_death(self):
        while self.alive:
            self.step()
        return self.compute_fitness()


# ── Genetic Algorithm ──────────────────────────────────────────────────────────
def _evaluate(brain):
    """Top-level so multiprocessing can pickle it. Returns (fitness, length)."""
    agent = SnakeAgent(brain)
    return agent.run_to_death(), agent.length


def tournament_select(population, fitnesses, k=TOURNAMENT_K):
    indices = random.sample(range(len(population)), k)
    best    = max(indices, key=lambda i: fitnesses[i])
    return population[best]


class GeneticAlgorithm:
    def __init__(self):
        self.generation   = 0
        self.population   = [NeuralNet() for _ in range(POPULATION_SIZE)]
        self.best_brain   = None   # type: Optional[NeuralNet]
        self.best_fitness = 0.0
        self.history      = []    # (gen, best_fit, avg_fit, max_len)
        self.pool         = Pool()   # persistent — avoids spin-up cost each gen

    def evolve(self):
        self.generation += 1

        results   = self.pool.map(_evaluate, self.population)
        fitnesses = np.array([f for f, _ in results], dtype=np.float64)
        lengths   = [l for _, l in results]

        best_idx = int(np.argmax(fitnesses))
        best_fit = float(fitnesses[best_idx])
        avg_fit  = float(fitnesses.mean())
        max_len  = max(lengths)

        if best_fit > self.best_fitness:
            self.best_fitness = best_fit
            self.best_brain   = self.population[best_idx]

        self.history.append((self.generation, best_fit, avg_fit, max_len))

        elite_n = max(2, int(POPULATION_SIZE * ELITE_FRACTION))
        ranked  = sorted(zip(fitnesses, self.population),
                         key=lambda x: x[0], reverse=True)
        elites  = [b for _, b in ranked[:elite_n]]

        inject_n = max(1, int(POPULATION_SIZE * RANDOM_INJECT))
        new_pop  = list(elites)
        fit_list = fitnesses.tolist()
        pop      = self.population

        while len(new_pop) < POPULATION_SIZE - inject_n:
            p1 = tournament_select(pop, fit_list)
            p2 = tournament_select(pop, fit_list)
            new_pop.append(p1.crossover(p2).mutate())

        new_pop += [NeuralNet() for _ in range(inject_n)]

        self.population = new_pop
        return self.generation, best_fit, avg_fit, max_len, self.best_brain


# ── Live Snake (visual replay — inherits all logic from SnakeAgent) ────────────
class LiveSnake(SnakeAgent):
    def render(self, surface):
        fx, fy = self.fruit
        pygame.draw.circle(surface, RED, (fx, fy), GRID_SIZE // 2)

        for idx, (x, y) in enumerate(self.positions):
            if idx == 0:
                pygame.draw.circle(surface, GREEN, (x, y), int(GRID_SIZE / 1.6))
                if self.direction in (UP, DOWN):
                    pygame.draw.circle(surface, BLACK, (x + 4, y), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (x - 4, y), GRID_SIZE // 4)
                else:
                    pygame.draw.circle(surface, BLACK, (x, y + 4), GRID_SIZE // 4)
                    pygame.draw.circle(surface, BLACK, (x, y - 4), GRID_SIZE // 4)
            else:
                t   = idx / max(len(self.positions), 1)
                col = (
                    int(10  + (1 - t) * (BLUE[0] - 10)),
                    int(100 + (1 - t) * (BLUE[1] - 100)),
                    int(200 + (1 - t) * (BLUE[2] - 200)),
                )
                pygame.draw.circle(surface, col, (x, y), GRID_SIZE // 2)


# ── HUD ────────────────────────────────────────────────────────────────────────
def draw_panel(screen, ga, gen, best_fit, avg_fit, max_len, snake):
    px = WIDTH + 8
    pygame.draw.rect(screen, PANEL_COLOR, (px, 0, PANEL_W, HEIGHT + 8))
    pygame.draw.line(screen, ACCENT, (px, 0), (px, HEIGHT + 8), 1)

    font_title = pygame.font.SysFont("consolas", 14, bold=True)
    font_body  = pygame.font.SysFont("consolas", 13)
    font_small = pygame.font.SysFont("consolas", 11)

    y = 16
    screen.blit(font_title.render("SNAKE  GA", True, ACCENT), (px + 12, y))
    y += 24
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    def row(label, val, color=OFF_WHITE):
        nonlocal y
        lbl = font_body.render(label, True, (120, 120, 140))
        v   = font_body.render(str(val), True, color)
        screen.blit(lbl, (px + 12, y))
        screen.blit(v,   (px + PANEL_W - v.get_width() - 12, y))
        y += 20

    row("Generation",   gen,                              YELLOW)
    row("Score",        snake.length,                     GREEN)
    row("Circle pen.",  f"{snake.circle_penalty:.0f}",    (255, 100, 100))
    row("Best Fit",     f"{best_fit:.0f}",                ACCENT)
    row("Avg Fit",      f"{avg_fit:.0f}")
    row("Best Length",  max_len,                          GREEN)
    row("Population",   POPULATION_SIZE)
    row("Tournament K", TOURNAMENT_K)
    row("Mutation",     f"{MUTATION_RATE * 100:.0f}%")

    y += 8
    pygame.draw.line(screen, (40, 40, 60), (px + 8, y), (px + PANEL_W - 8, y), 1)
    y += 10

    screen.blit(font_small.render("FITNESS HISTORY", True, (100, 100, 120)), (px + 12, y))
    y += 14

    history = ga.history[-20:]
    if len(history) > 1:
        bests = [h[1] for h in history]
        avgs  = [h[2] for h in history]
        max_v = max(bests) if max(bests) > 0 else 1
        gw, gh = PANEL_W - 24, 60
        gx, gy = px + 12, y
        pygame.draw.rect(screen, (25, 25, 40), (gx, gy, gw, gh))

        def plot_line(vals, color):
            pts = [(gx + int(i / (len(vals) - 1) * gw),
                    gy + gh - int((v / max_v) * gh))
                   for i, v in enumerate(vals)]
            if len(pts) > 1:
                pygame.draw.lines(screen, color, False, pts, 1)

        plot_line(avgs,  (60, 80, 120))
        plot_line(bests, ACCENT)
        y += gh + 8

    status_color = GREEN if snake.alive else RED
    screen.blit(font_body.render("ALIVE" if snake.alive else "DIED", True, status_color),
                (px + 12, y))
    screen.blit(font_small.render("SPACE=skip  F=speed  Q=quit", True, (70, 70, 90)),
                (px + 12, HEIGHT - 20))


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    total_w = WIDTH + 8 + PANEL_W
    screen  = pygame.display.set_mode((total_w, HEIGHT + 8))
    pygame.display.set_caption("Snake — Neural Net Genetic Algorithm")
    surface = pygame.Surface((WIDTH + 8, HEIGHT + 8)).convert()
    clock   = pygame.time.Clock()

    ga = GeneticAlgorithm()

    SPEEDS    = [5, 10, 20, 40, 80]
    speed_idx = 1

    print(f"Snake GA  |  pop={POPULATION_SIZE}  net=29->24->16->3  parallel=ON")
    print(f"Revisit penalty={REVISIT_PENALTY}  visit decay={VISIT_DECAY}  "
          f"max_no_eat={MAX_STEPS_NO_EAT}  BFS reachability=ON")
    print("-" * 55)

    while True:
        print(f"Gen {ga.generation + 1}: training {POPULATION_SIZE}...",
              end=" ", flush=True)
        gen, best_fit, avg_fit, max_len, best_brain = ga.evolve()
        print(f"best={best_fit:.0f}  avg={avg_fit:.0f}  max_len={max_len}")

        snake = LiveSnake(best_brain)
        skip  = False

        while snake.alive and not skip:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_SPACE:
                        skip = True
                    elif event.key == pygame.K_f:
                        speed_idx = (speed_idx + 1) % len(SPEEDS)

            snake.step()

            surface.fill(GREY)
            for gx in range(0, WIDTH, GRID_SIZE):
                for gy in range(0, HEIGHT, GRID_SIZE):
                    pygame.draw.circle(surface, (40, 40, 40),
                                       (gx + GRID_SIZE // 2, gy + GRID_SIZE // 2), 1)
            snake.render(surface)

            screen.fill(DARK_GREY)
            screen.blit(surface, (0, 0))
            draw_panel(screen, ga, gen, best_fit, avg_fit, max_len, snake)

            pygame.display.flip()
            clock.tick(SPEEDS[speed_idx] + math.ceil(snake.length / 5) * 2)

        pygame.time.wait(400)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                pygame.quit(); sys.exit()


if __name__ == "__main__":
    main()
