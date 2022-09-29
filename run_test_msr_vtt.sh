#!/usr/bin/env zsh
CUDA_VISIBLE_DEVICES=0 python evaluate.py --dataset=msr-vtt --model=RMN \
 --result_dir=results/msrvtt/with_TRM_add_text_mem --attention=soft \
 --use_loc --use_rel --use_func \
 --hidden_size=1024 --att_size=1024 \
 --test_batch_size=32 --beam_size=2 \
 --eval_metric=CIDEr --topk=18 --max_words=26