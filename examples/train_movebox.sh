for seed in 111 222 333 444 555
do
    python train.py --algo med --env gridworld --task MoveBox --map multi_exits --exp_name performance --t_max 2000000 --horizon 50 --n_episodes 2 --use_wandb True --seed $seed
done
