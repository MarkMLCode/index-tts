@echo off
setlocal

uv run python tools/preprocess_multiproc.py ^
    --manifest "%INDEXTTS_FINETUNE_MANIFEST%" ^
    --output-dir "_processed_data\%INDEXTTS_FINETUNE_DATASET%" ^
    --model-dir checkpoints ^
    --config checkpoints/config.yaml ^
    --gpt-checkpoint checkpoints/gpt.pth ^
    --language en ^
    --device cuda ^
    --batch-size 1 ^
    --workers 0 ^
    --num-processes 1 ^
    --skip-existing ^
    --val-ratio 0.02

endlocal
