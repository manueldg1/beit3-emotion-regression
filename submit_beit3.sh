#!/bin/bash
#SBATCH --job-name=ft_beit3_A6000
#SBATCH --time=14-00:00:00
#SBATCH --mem=60G
#SBATCH --partition=hlt_msc
#SBATCH --gres=gpu:A6000_48GB:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/cfs/home/u036743/unilm/beit3/output/va_results/logs/ft_beit3_%j.out

# Variabili d'ambiente per il training distribuito (anche su GPU singola)
export MASTER_ADDR=localhost
export MASTER_PORT=$(shuf -i 20000-30000 -n 1)
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

# Path al tuo environment specifico
PYTHON_BIN="/cfs/home/u036743/miniconda3/envs/emotion_g06/bin/python"
cd "/cfs/home/u036743/unilm/beit3"

# Lancio del training
$PYTHON_BIN -u run_beit3_finetuning.py \
--task va_regression \
--model beit3_large_patch16_480_valence_arousal \
--data_path /cfs/home/u036743/emotion_dataset/processed_data \
--finetune /cfs/home/u036743/unilm/beit3/Multilingual_projection/xlmr_fixed.pt \
--resume /cfs/home/u036743/unilm/beit3/output/va_results/checkpoint-best.pth \
--output_dir "./output/va_results" \
--batch_size 4 \
--update_freq 16 \
--checkpoint_activations \
--input_size 480 \
--lr 1e-5 \
--epochs 10 \
--sentencepiece_model /cfs/home/u036743/unilm/beit3/beit3_xlm_sentencepiece.bpe.model \
--num_max_bpe_tokens 64 \
--nb_classes 2 \
--num_workers 8 \
--clip_grad 1.0
