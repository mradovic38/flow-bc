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
**Half Cheetah** | medium-v0 | 15043.4619 | **15491.5459**
**Pusher** | medium-v0 |
**Hopper** | medium-v0 |
**Humanoid** | medium-v0 | 7789.0345 | **8100.7404**
**Inverted Pendulum** | medium-v0 |
**Inverted Double Pendulum** | medium-v0 |
**Swimmer** | medium-v0 |
**Walker2d** | medium-v0 |
**Ant** | medium-v0 |
**Reacher** | medium-v0 |


### D4RL
Env | Dataset | Gaussian | Flow-Matching (ours)
|----------|----------|----------|----------|
**Door** | human-v2 | 168.7029 | **207.0339**
**Kitchen** | mixed-v2 | 606.4000 | **723.7400**
**Pen** | human-v2 |
**Hammer** | human-v2 |
**Relocate** | human-v2 |
**Ant Maze** | 
**Point Maze** |