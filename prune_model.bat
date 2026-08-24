@echo off
setlocal

uv run python tools/prune_gpt_checkpoint.py ^
    --input "_trained\%INDEXTTS_FINETUNE_DATASET%\latest.pth" ^
    --output "_trained\%INDEXTTS_FINETUNE_DATASET%_indextts_2_5.pth"

endlocal
