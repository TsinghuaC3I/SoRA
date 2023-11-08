cd ..
export WANDB_DISABLED=false
for task in rte # can be cola/mrpc/rte/stsb/qnli/sst2/qqp/mnli-m/mnli-mm
do
lambda=0.001
lambda2=0
max_lambda=7e-4
lambda_num=7
lr=1.2e-3
r=8
epoch=50
seed=48
bsz=32
epoch2=15
echo $task
echo "lambda=" $lambda
echo $seed
CUDA_VISIBLE_DEVICES=0 \
python -u run_glue.py \
    --do_eval \
    --do_predict \
    --do_train \
    --task_name $task \
    --eval_steps 1000 \
    --evaluation_strategy steps \
    --greater_is_better true \
    --learning_rate $lr \
    --max_grad_norm 0.1 \
    --load_best_model_at_end \
    --logging_steps 100 \
    --max_steps -1 \
    --model_name_or_path /root/xtlv/data/models/DeBERTaV3_base \
    --num_train_epochs $epoch \
    --output_dir results/$task-lambda2_${lambda2}_${max_lambda}_lambda_${lambda}_epoch_${epoch}_seed_${seed}_${epoch2} \
    --overwrite_output_dir \
    --per_device_eval_batch_size $bsz \
    --per_device_train_batch_size $bsz \
    --save_steps 1000 \
    --save_strategy steps \
    --save_total_limit 1 \
    --tokenizer_name /root/xtlv/data/models/DeBERTaV3_base \
    --warmup_ratio 0.06 \
    --warmup_steps 0 \
    --weight_decay 0.1 \
    --disable_tqdm true \
    --load_best_model_at_end \
    --ddp_find_unused_parameters false \
    --sparse_lambda $lambda \
    --sparse_lambda_2 $lambda2 \
    --seed $seed \
    --lora_r $r \
    --max_seq_length 320 \
    --max_lambda $max_lambda \
    --lambda_schedule linear \
    --lambda_num $lambda_num \
    --train_sparse > results/$task-lambda2_${lambda2}_${max_lambda}_lambda_${lambda}_epoch_${epoch}_seed_${seed}_${epoch2}.log 2>&1
wait
done