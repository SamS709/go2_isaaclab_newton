## go2_isaaclab_newton

The main objectives are:

- **Train reinforcement learning policies** for the Go2 robot in simulation.
- **Sim2Sim transfer**: Validate and adapt policies trained in PhysX for use in the Newton engine, enabling robust transfer between simulation environments.

This work is part of a larger project aimed at validating trained policies for legged robots in diverse simulation environments. For more details and the broader context, see the main project repository: [Main Project GitHub](https://github.com/SamS709/go2_isaaclab.git)  <!-- Replace with actual link -->


<img src="images/Newton_MuJoCo.png" width="400"/>

*sim to sim from PhysX to Newton*


## Installation

**Setup Requirements**

To use this repository, you must be in a Python environment where the [`feature/newton`](https://github.com/isaac-sim/IsaacLab/tree/feature/newton) branch of IsaacLab has been installed.

Here are the [`detailed steps`](https://isaac-sim.github.io/IsaacLab/main/source/experimental-features/newton-physics-integration/installation.html) to install it. Using a conda env is highly recommended.

**Repo install**

```bash
git clone https://github.com/SamS709/go2_isaaclab_newton.git
cd go2_isaaclab_newton
python -m pip install -e source/go2_isaaclab_newton
```


## Training Go2 Robot Policies

To train a policy for the Go2 robot using IsaacLab, run:

```bash
python scripts/rsl_rl/train.py --task Isaac-Velocity-Go2-Asymmetric-v0 --num_envs 4096 --headless
```

## Sim2Sim Transfer (PhysX to Newton)

To perform sim2sim transfer and validate a trained policy from PhysX in the Newton engine, use:

```bash
python scripts/sim2sim_transfer/rsl_rl_transfer.py \
    --task=Isaac-Velocity-Go2-Asymmetric-v0 \
    --num_envs=32 \
    --checkpoint scripts/sim2sim_transfer/checkpoint.pt \
    --policy_transfer_file scripts/sim2sim_transfer/config/physx_to_newton_go2.yaml \
    --visualizer newton
```
