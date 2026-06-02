# Cyberpunk BCI Connect-4

This project is a Cyberpunk-themed Connect-4 game that integrates Brain-Computer Interface (BCI) simulation, Convolutional Neural Networks (CNN), and Monte Carlo Tree Search (MCTS). It features a real-time win-rate predictor and an asynchronous AI calculation system to ensure smooth gameplay.

## Features

- Cyberpunk UI: Developed using Pygame with dynamic scaling and high-contrast visual effects.
- BCI Integration: Includes a MockBCI simulator that processes focus, relaxation, and blink signals.
- AI Logic: Combines CNN-based board evaluation with MCTS and minimax search.
- Data Pipeline: Automated self-play data generation and ResNet model training.

## Installation

Ensure you have Python 3.8 or higher installed. This project uses `uv` for fast, reliable package and environment management. 

Install the necessary dependencies using `uv`:
```bash
uv venv

uv pip install -r requirements.txt

uv run python connect4.py