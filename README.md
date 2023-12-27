# Multi-Expert Distillation for Few-Shot Coordination (Student Abstract)

This repository contains the implementation of Multi-Expert Distillation (MED), based on PyTorch. 

## 1. Getting started

Use the install script to install the python environment:

```shell
bash install.sh
conda activate med
```

## 2. Run an experiment
All the experiments can be run with the unified entrance file `examples/train.py` with customized arguments.

### LIPO
The repository consists of a re-implementation of [LIPO]([https://sites.google.com/view/iclr-lipo-2023).
For generating a population in Girdworld MoveBox or Overcooked, enter the `examples` folder and run the following command:
```bash
python train.py --algo lipo --env gridworld --task MoveBox --map multi_exits --exp_name test --use_wandb True --pop_size 8 --horizon 50 --n_iter 500 --eval_interval 10 --n_sp_ts 5000 --n_xp_ts 5000 --eval_interval 10
```
```bash
python train.py --algo lipo --env overcooked --map_name full_divider_salad_multi_ingred --exp_name test --use_wandb True --pop_size 8 --horizon 100 --n_iter 1000 --n_sp_ts 5000 --n_xp_ts 5000 --eval_interval 10
```
The results and models can be found in the `examples/results` folder. 
### MED
To run MED, the population model files should be placed in the `harl/runners/generalist_runners/models` folder. Users should make sure the file is named properly. 
For running MED, enter the `examples` folder and run the following commands:
```bash
python train.py --algo med --env matrix_game --exp_name performance --t_max 30000 --n_episodes 3 --use_wandb True
```
```bash
python train.py --algo med --env gridworld --task MoveBox --map multi_exits --exp_name performance --t_max 2000000 --horizon 50 --n_episodes 2 --use_wandb True
```
```bash
python train.py --algo med --env overcooked --map_name full_divider_salad_multi_ingred --exp_name performance --t_max 7500000 --horizon 100 --n_episodes 2 --use_wandb True
```
Training scripts are also provided in the `examples` folder.
