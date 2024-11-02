

log_lrs=(-10 -8 -6 -4 -2)
width=(128 512 2048)
repalambda=(0.0 1.0)
export TOKENIZERS_PARALLELISM=false

for loglr in "${log_lrs[@]}"; do
    for w in "${width[@]}"; do
        for repa_lambda in "${repalambda[@]}"; do
            lr=$(python -c "print(2**$loglr)")
            echo "Running with lr=$lr and width=$w"
            name="lr${lr}_width${w}_repalambda${repa_lambda}"

            torchrun --nproc_per_node=8 trainer.py \
                --run_name $name \
                --model_width $w \
                --learning_rate $lr \
                --dataset_url "/home/ubuntu/pd12m.int8/dataset/cc12m-wds/cc12m-train-{0000..2151}.tar" \
                --test_dataset_url "/home/ubuntu/pd12m.int8/dataset/cc12m-wds/cc12m-train-{2152..2168}.tar" \
                --num_epochs 2 \
                --batch_size 64 \
                --max_steps 5000 \
                --evaluate_every 1000 \
                --alignment_layer 8 \
                --repa_lambda $repa_lambda \
                --model_depth 9 \
                --model_head_dim 32
        done
    done
done