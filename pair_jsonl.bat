@echo off
setlocal

uv run python tools/generate_gpt_pairs.py ^
    --dataset "_processed_data\%INDEXTTS_FINETUNE_DATASET%" ^
    --pairs-per-target 2 ^
    --force

endlocal
