for seed in 111 222 333 444 555
do
    python train.py --algo med --env matrix_game --exp_name performance --use_wandb True --seed $seed --t_max 30000
done
