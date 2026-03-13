CUDA_VISIBLE_DEVICES=2 python3 -u run_beit3_finetuning.py \
--task va_regression \
--model beit3_large_patch16_480_valence_arousal \
--vocab_size 250002 \
--data_path /cfs/home/u036743/emotion_dataset/processed_data \
--finetune /cfs/home/u036743/unilm/beit3/Multilingual_projection/xlmr_fixed.pt \
--output_dir "./output/test_loss_MSE" \
--log_dir "./output/test_loss_MSE/logs" \
--batch_size 16 \
--update_freq 2 \
--input_size 480 \
--lr 2e-5 \
--warmup_epochs 1 \
--clip_grad 1.0 \
--epochs 5 \
--sentencepiece_model /cfs/home/u036743/unilm/beit3/beit3_xlm_sentencepiece.bpe.model \
--num_max_bpe_tokens 64 \
--nb_classes 2 \
--num_workers 8 \
















CUDA_VISIBLE_DEVICES=4 python3 -u run_beit3_finetuning.py \
--task va_regression \
--model beit3_large_patch16_480_valence_arousal \
--vocab_size 250002 \
--data_path /cfs/home/u036743/emotion_dataset/processed_data \
--finetune /cfs/home/u036743/unilm/beit3/Multilingual_projection/xlmr_fixed.pt \
--output_dir "./output/test_robust" \
--log_dir "./output/test_robust/logs" \
--loss_type robust \
--batch_size 8 \
--update_freq 4 \
--input_size 480 \
--lr 2e-5 \
--warmup_epochs 1 \
--clip_grad 0.5 \
--epochs 5 \
--sentencepiece_model /cfs/home/u036743/unilm/beit3/beit3_xlm_sentencepiece.bpe.model \
--num_max_bpe_tokens 64 \
--nb_classes 2 \
--num_workers 8
