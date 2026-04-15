# Flow Matching Policy for Behavioral Cloninig

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
    for wandb sweeps.

6. Test
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
**Humanoid** | medium-v0 | 7581.16 ± 1838.37 | **8213.03 ± 36.86**
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
**Kitchen** | mixed-v2 | 603.38 ± 234.78 | **740.16 ± 151.83**
**Pen** | human-v2 |
**Hammer** | human-v2 |
**Relocate** | human-v2 |
**Ant Maze** | 
**Point Maze** |


## Ablation study on Integration Steps
### Humanoid-medium-v0
ODE steps | Avg Return ± Std | Latency (ms)
|----------|----------|----------|
1            | 7031.33 ± 2263.08         | 0.11     
2            | 7642.93 ± 1824.08         | 0.16           
4            | 7957.36 ± 1214.06         | 0.29           
8            | 7836.65 ± 1445.65         | 0.53           
10           | 8172.81 ± 574.24          | 0.65           
12           | 8213.03 ± 36.86           | 0.78           
16           | 7871.77 ± 1317.44         | 1.03           
20           | 7979.39 ± 1233.45         | 1.26           
24           | 7973.92 ± 1186.77         | 1.50           
36           | 8061.47 ± 967.44          | 2.21           
50           | 8034.26 ± 1178.72         | 3.09    

### Kitchen-mixed-v2
ODE steps | Avg Return ± Std | Latency (ms)
|----------|----------|----------|
1            | 17.35 ± 61.87           |  0.13         
2            | 17.78 ± 78.61           |  0.20         
4            | 594.98 ± 236.30          | 0.34          
8            | 674.19 ± 161.76          | 0.62          
10           | 611.45 ± 286.67          | 0.75            
12           | 539.10 ± 295.77          | 0.91           
16           | 740.16 ± 151.83          | 1.18           
20           | 681.41 ± 213.52          | 1.45           
24           | 707.02 ± 206.12          | 1.71           
36           | 722.46 ± 172.70          | 2.58           
50           | 730.37 ± 166.26          | 3.51                 


