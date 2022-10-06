#!/usr/bin/env zsh
CUDA_VISIBLE_DEVICES=4 python train.py --dataset=msr-vtt --model=RMN \
            --result_dir=results/msrvtt/full_model_64_query_with_orthogonal_ --use_lin_loss \
            --learning_rate_decay --learning_rate_decay_every=5 \
            --learning_rate_decay_rate=3 \
            --use_loc --use_rel --use_func \
            --learning_rate=1e-4 --attention=soft \
            --hidden_size=1024 --att_size=1024 \
            --train_batch_size=64 \
            --test_batch_size=32 \
            --beam_size=2