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

4. Generate [wandb](https://wandb.ai/) API key, create .env file and add `WANDB_API_KEY`
    ```
    WANDB_API_KEY=<YOUR_API_KEY>
    ```

5. Train
    ```sh
    uv run python train.py --config config/halfcheetah_gaussian_baseline.yaml
    ```
    for regular training with wandb.

    OR

    ```sh
    uv run python train.py --config config/halfcheetah_gaussian_baseline.yaml --disable-wandb
    ```
    for regular training without wandb.

    OR

    ```sh
    wandb sweep config/halfcheetah_gaussian_baseline.yaml       
    wandb agent <AGENT_NAME>
    ```   
    for hyper-parameter tuning

1. Test
    ```sh
    uv run python test.py --config config/halfcheetah_gaussian_baseline.yaml --video-dir videos --num-episodes 5
    ```