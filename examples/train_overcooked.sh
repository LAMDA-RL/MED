for seed in 111 222 333 444 555
do
    python train.py --algo med --env overcooked --map_name full_divider_salad_multi_ingred --exp_name performance --t_max 7500000 --horizon 100 --n_episodes 2 --use_wandb True --seed $seed
done
