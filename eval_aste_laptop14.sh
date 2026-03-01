set -ex

export CUDA_VISIBLE_DEVICES=0

declare -A TASK_DATA

# TASK_DATA[aste]="laptop14 rest14 rest15 rest16"
TASK_DATA[aste]="laptop14"

cd src


for TASK in aste
do
for DATA in ${TASK_DATA[${TASK}]}
do
for DATA_RATIO in 1.0
do

for SEED in 151
do
for K in 6

do
INFER_PATH=$K
CTRL_TOKEN=post
OUT_DIR="../output/$TASK/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${SEED}"

# 测试结果输出目录（和训练目录分离，避免污染）
TEST_OUTPUT_DIR="../test_output/${TASK}/${DATA}/top${K}_${CTRL_TOKEN}_data${DATA_RATIO}_seed${SEED}"

mkdir -p $OUT_DIR
mkdir -p $TEST_OUTPUT_DIR

/usr/local/miniconda3/bin/python3 main.py \
    --data_path "../data/" \
    --dataset $DATA \
    --model_name_or_path "$OUT_DIR/final/" \
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
    --do_inference \
    | tee ${TEST_OUTPUT_DIR}/test.log \
    2> ${TEST_OUTPUT_DIR}/test.err
    # --model_name_or_path "PATH TO THE CHECKPOINT" \ # configure the checkpoint path to eval


done
done
done
done
done
# done
