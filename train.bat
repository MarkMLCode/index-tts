@echo off
setlocal

uv run python trainers/train_gpt_v2.py ^
    --train-manifest "_processed_data\%INDEXTTS_FINETUNE_DATASET%\gpt_pairs_train.jsonl" ^
    --val-manifest "_processed_data\%INDEXTTS_FINETUNE_DATASET%\gpt_pairs_val.jsonl" ^
    --config checkpoints/config.yaml ^
    --base-checkpoint checkpoints/gpt.pth ^
    --output-dir "_trained\%INDEXTTS_FINETUNE_DATASET%" ^
    --batch-size 2 ^
    --grad-accumulation 2 ^
    --epochs 20 ^
    --learning-rate 1e-5 ^
    --weight-decay 0.01 ^
    --warmup-steps 1000 ^
    --log-interval 10 ^
    --val-interval 1000 ^
    --save-every 1000 ^
    --grad-clip 1.0 ^
    --text-loss-weight 0.2 ^
    --mel-loss-weight 0.8 ^
    --amp ^
    --amp-dtype auto

endlocal
