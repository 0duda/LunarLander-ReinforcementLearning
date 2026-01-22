# Lunar Lander - Reinforcement Learning Project

This project implements Reinforcement Learning agents for the **Lunar Lander** environment from Gymnasium, using the **PPO (Proximal Policy Optimization)** algorithm from Stable-Baselines3. It project was developed as part of the Introduction of Intelligent and Autonomous Systems course at FCUP (BSc in Artificial Intelligence and Data Science, University of Porto, 2025/2026). The goal is to train RL agents to perform successful landings in the Lunar Lander environment, with different reward variations and constraints.

### Authors

- Carolina Proença
- Eduarda Neves
- Maria Morais

### Main Features

- **Environment**: Gymnasium Lunar Lander (continuous mode)
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Reward Shaping**: Custom implementations for soft landing, precision landing, and combinations
- **Constraints**: Various tested constraints (wind, random force, moving helipad, etc.)

## Project Structure

```
LunarLander-ReinforcementLearning/
├── train.py                           # Main training script
├── visual.py                          # Agent visualization script
├── evaluation_of_constraints.py       # Constraint evaluation
├── soft_precision_landing_no_wind.py  # Main custom environment
├── constraint_*.py                    # Constraint implementations
├── models/                            # Trained models
│   ├── Baseline/
│   ├── Soft+Precision/
│   ├── Soft+Precision+CNN/
│   └── ...
├── logs/                              # TensorBoard logs
└── README.pdf                         # Detailed project documentation (for submission)
```

## Requirements

- Python 3.8+
- Gymnasium
- Stable-Baselines3
- Box2D
- NumPy
- TensorBoard (for log visualization)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/your-username/submission_lunarlander.git
cd submission_lunarlander
```

2. Install dependencies:
```bash
pip install gymnasium[box2d]
pip install stable-baselines3[extra]
pip install tensorboard
```

## How to Use

### Train a New Model

1. Edit the `train.py` file to choose the desired environment:
```python
from soft_precision_landing_no_wind import LunarLander  # Choose the environment
```

2. Configure the output directory:
```python
models_dir = "models/YourModel"  # Your model name
```

3. Run the training:
```bash
python train.py
```

### Visualize a Trained Agent

1. Edit `visual.py` to load the desired model
2. Run:
```bash
python visual.py
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=./logs
```

Access `http://localhost:6006` in your browser.

## Available Models

The project includes several trained models with different configurations:

| Model              | Description |
|--------|-----------|
| **Baseline** | Base model without modifications |
| **Baseline+CNN** | Baseline with CNN network |
| **Feet** | Focus on lander feet control |
| **HorizontalDescent** | Optimized for horizontal descent |
| **PrecisionLanding** | Focus on landing at (0,0) |
| **Soft+Precision** | Combines soft landing and precision |
| **Soft+Precision+CNN** | CNN version of the above |
| **Soft+Precision+1M** | Trained with 1M timesteps |
| **Soft+Precision+Bandeiras** | With flags system |

## Implemented Constraints

The project explores different constraints to test agent robustness:

- `constraint_wind.py` - Adds wind to the environment
- `constraint_random_force.py` - Applies random forces
- `constraint_moving_helipad.py` - Moving helipad
- `constraint_random_inicial_x.py` - Random initial X position

To evaluate a model with constraints:
```bash
python evaluation_of_constraints.py
```

## Results Visualization

### TensorBoard

Training logs are saved in `./logs` and can be visualized with TensorBoard:
```bash
tensorboard --logdir=./logs
```

### Available Metrics

- Episode reward
- Episode length
- Policy loss
- Value loss
- Entropy
- Learning rate

## Useful Links

- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
