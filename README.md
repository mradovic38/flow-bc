# Flow-Matching Policy Behavioral Cloninig

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

4. Generate [wandb](https://wandb.ai/) API key, create .env file and add `WANDB_API_KEY`
    ```
    WANDB_API_KEY=<YOUR_API_KEY>
    ```

5. Train
    ```sh
    uv run python train.py --config config/halfcheetah/gaussian_baseline.yaml
    ```
    for regular training with wandb.

    OR

    ```sh
    uv run python train.py --config config/halfcheetah/gaussian_baseline.yaml --disable-wandb
    ```
    for regular training without wandb.

    OR

    ```sh
    wandb sweep config/halfcheetah/gaussian_baseline.yaml       
    wandb agent <AGENT_NAME>
    ```   
    for hyper-parameter tuning

1. Test
    ```sh
    uv run python test.py --config config/halfcheetah/gaussian_baseline.yaml --video-dir videos/halfcheetah --num-episodes 5
    ```


## Average return in 100 episodes

### Mujoco
Env | Dataset | Gaussian | Flow-Matching (ours)
|----------|----------|----------|----------|
**Half Cheetah** | medium-v0 | 14871.47 ± 3046.86 | **15499.46 ± 51.01**
**Pusher** | medium-v0 |
**Hopper** | medium-v0 | 3577.98 ± 29.51 | **3593.45 ± 31.59**
**Humanoid** | medium-v0 | 7581.16 ± 1838.37 | **8108.63 ± 807.94**
**Inverted Pendulum** | medium-v0 |
**Inverted Double Pendulum** | medium-v0 |
**Swimmer** | medium-v0 | **274.17 ± 19.95** | 227.50 ± 12.08
**Walker2d** | medium-v0 | **6235.86 ± 29.79** | 6204.74 ± 79.81
**Ant** | medium-v0 | 5769.32 ± 1602.93 | **6027.36 ± 1050.83**
**Reacher** | medium-v0 |


### D4RL
Env | Dataset | Gaussian | Flow-Matching (ours)
|----------|----------|----------|----------|
**Door** | human-v2 | 172.22 ± 179.17 | **198.01 ± 289.67**
**Kitchen** | mixed-v2 | 603.38 ± 234.78 | **707.50 ± 171.30**
**Pen** | human-v2 |
**Hammer** | human-v2 |
**Relocate** | human-v2 |
**Ant Maze** | 
**Point Maze** |