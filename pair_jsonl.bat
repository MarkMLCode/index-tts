@echo off
setlocal

uv run python tools/generate_gpt_pairs.py ^
    --dataset "%INDEXTTS_FINETUNE_DATASET%_processed_data" ^
    --pairs-per-target 2 ^
    --force

endlocal
