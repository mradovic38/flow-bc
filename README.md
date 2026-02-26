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

4. Train
```sh
uv run python train.py
```

5. Test
```sh
uv run python test.py
```