cd ..
for lora_r in 16
do
for seed in 100
do
task=mnli-mm # can be cola/mrpc/rte/stsb/qnli/sst2/qqp/mnli-m/mnli-mm
lr=3e-4
bsz=8 # rte: 32; the other: 8
epoch=10
echo $lr
echo $seed
echo $bsz
echo $task
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
    --output_dir results/${task}_lora_r_${lora_r}_lr_${lr}_bsz_${bsz}_epoch_${epoch}_seed_${seed} \
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
    --sparse_lambda 0 \
    --seed $seed \
    --lora_r $lora_r \
    --max_seq_length 256 > results/${task}_lora_r_${lora_r}_lr_${lr}_bsz_${bsz}_epoch_${epoch}_seed_${seed}.log 2>&1
done
done
