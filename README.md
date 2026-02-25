# Behavioral Cloninig

## Setup

1. Install UV
```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies
```sh
uv sync
```

3. Download Half Cheetah Dataset
```sh
minari download mujoco/halfcheetah/medium-v0    
```

4. Run main.py
```sh
uv run python main.py
```