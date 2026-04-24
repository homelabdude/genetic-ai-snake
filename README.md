# Genetic AI Plays Snake

A genetic algorithm that evolves a neural network to (sort of) play Snake on a 40×40 wrapping grid.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
python snake_v6.py               # latest version — GA training with curriculum
python snake_v0.py               # just the game, playable manually
```

**Controls:** `WASD` or arrow keys (manual mode) · `SPACE` skip generation · `F` cycle speed · `Q` quit

## Versions

| File | Description |
|------|-------------|
| `snake_v0.py` | Playable game, neural net wired up but no training |
| `snake_v1.py` | First GA — 500 population, basic fitness function |
| `snake_v2.py` | Circle penalties, larger network, population 1000 |
| `snake_v3.py` | Parallelised evaluation, tournament selection, extra fruit direction inputs |
| `snake_v4.py` | BFS reachability input, dynamic step budget, progress reward |
| `snake_v5.py` | Curriculum learning + 3 fruits — best run hit length 182 |
| `snake_v6.py` | Performance-based curriculum graduation, 1 fruit, honest ceiling ~105 |

## Read the write-up

Full devlog with results and commentary: [here](https://homelabdude.com/posts/ai-learns-to-play-snake/)
