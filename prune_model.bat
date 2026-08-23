@echo off
setlocal

uv run python tools/prune_gpt_checkpoint.py ^
    --input "trained_ckpts_%INDEXTTS_FINETUNE_DATASET%/latest.pth" ^
    --output "models/%INDEXTTS_FINETUNE_DATASET%_indextts_2_5.pth"

endlocal
