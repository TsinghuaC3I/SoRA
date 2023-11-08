cd ..
export WANDB_DISABLED=false
for task in stsb # can be cola/mrpc/rte/stsb/qnli/sst2/qqp/mnli-m/mnli-mm
do
for lora_r in 8
do
for lambda in 10
do
for lambda2 in 3e-4
do
for seed in 0 21 42 81 100
do
for lr in 8e-4
do
epoch=20
bsz=8
echo $task
echo "lambda=" $lambda
echo "lambda2=" $lambda2
echo "lora_r=" $lora_r
echo "seed=" $seed
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
    --output_dir results/${task}_lora_r_${lora_r}_lambda_${lambda}_lambda2_${lambda2}_lr_${lr}_epoch_${epoch}_bsz_${bsz}_seed_${seed} \
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
    --lora_r $lora_r \
    --max_seq_length 128 \
    --train_sparse > results/${task}_lora_r_${lora_r}_lambda_${lambda}_lambda2_${lambda2}_lr_${lr}_epoch_${epoch}_bsz_${bsz}_seed_${seed}.log 2>&1
wait
done
done
done
done
done
done