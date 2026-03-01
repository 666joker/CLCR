# export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64  # 调小内存块拆分大小，减少碎片

set -ex

export CUDA_VISIBLE_DEVICES=0

declare -A TASK_DATA

# TASK_DATA[aste]="laptop14 rest14 rest15 rest16"
TASK_DATA[aste]="rest16"

cd src

for TASK in aste
do
for DATA in ${TASK_DATA[${TASK}]}
do
for DATA_RATIO in 1.0
do
# for SEED in 5 10 15 20 25
for SEED in 5
do
for K in 6

do
INFER_PATH=$K
CTRL_TOKEN=post
OUT_DIR="../outputs/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${SEED}"

mkdir -p $OUT_DIR


/usr/local/miniconda3/bin/python3 main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name_or_path "../t5-base-local/" \
    --output_dir $OUT_DIR \
    --num_train_epochs 20 \
    --save_top_k 0 \
    --task $TASK \
    --top_k $K \
    --ctrl_token $CTRL_TOKEN \
    --multi_path \
    --num_path $INFER_PATH \
    --seed $SEED \
    --train_batch_size 16 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lowercase \
    --sort_label \
    --data_ratio $DATA_RATIO \
    --check_val_every_n_epoch 10  \
    --agg_strategy consensus \
    --eval_batch_size 16 \
    --constrained_decode \
    --do_train \
    | tee ${OUT_DIR}/train.log \
    2> ${OUT_DIR}/train.err
    # --model_name_or_path "PATH TO THE CHECKPOINT" \ # configure the checkpoint path to eval

done
done
done
done
done
# done
