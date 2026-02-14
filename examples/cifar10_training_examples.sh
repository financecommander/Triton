#!/bin/bash
# CIFAR-10 Training Examples
# 
# This script contains ready-to-use commands for common CIFAR-10 training scenarios.
# Copy and paste the command you need, or run this script with the appropriate scenario number.

set -e

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

function print_command() {
    echo -e "${GREEN}Command:${NC}"
    echo "$1"
    echo ""
}

# Check if a scenario number was provided
if [ $# -eq 0 ]; then
    echo "CIFAR-10 Training Script Examples"
    echo ""
    echo "Usage: $0 <scenario_number>"
    echo ""
    echo "Available scenarios:"
    echo "  1. Fresh 500-epoch training run with all enhancements"
    echo "  2. Resume from epoch 76 to 500 epochs"
    echo "  3. Conservative training with minimal augmentation"
    echo "  4. Aggressive regularization (for overfitting)"
    echo "  5. Quick 100-epoch test run"
    echo "  6. Training with AutoAugment"
    echo "  7. Multi-experiment comparison setup"
    echo ""
    echo "Or simply copy-paste commands from this file!"
    exit 0
fi

SCENARIO=$1

case $SCENARIO in
    1)
        print_header "Scenario 1: Fresh 500-Epoch Training Run"
        echo "Best for: Starting a new long training run from scratch"
        echo "Features: CutMix, label smoothing, early stopping, comprehensive logging"
        echo ""
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --cutmix_alpha 1.0 \
    --save_freq 10 \
    --log_interval 100 \
    --workers 4"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    2)
        print_header "Scenario 2: Resume from Epoch 76 to 500"
        echo "Best for: Continuing training from your current checkpoint"
        echo "Note: Update --resume path to your actual checkpoint file"
        echo ""
        
        # Check if checkpoint exists
        CHECKPOINT="./checkpoints/resnet18_cifar10_epoch_76.pth"
        if [ ! -f "$CHECKPOINT" ]; then
            echo "ERROR: Checkpoint not found: $CHECKPOINT"
            echo "Please specify the correct checkpoint path."
            echo ""
            echo "Available checkpoints:"
            ls -lh ./checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found in ./checkpoints/"
            exit 1
        fi
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume $CHECKPOINT \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 20 \
    --csv_log ./results/resume_to_500.csv"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    3)
        print_header "Scenario 3: Conservative Training"
        echo "Best for: Baseline run with standard augmentation"
        echo "Features: Only RandomCrop + HorizontalFlip, no advanced augmentation"
        echo ""
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 40 \
    --save_freq 20"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    4)
        print_header "Scenario 4: Aggressive Regularization"
        echo "Best for: When you see significant overfitting (train >> val acc)"
        echo "Features: CutMix + MixUp, high label smoothing, high weight decay"
        echo ""
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 1e-3 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.15 \
    --cutmix \
    --cutmix_alpha 1.0 \
    --mixup \
    --mixup_alpha 0.5 \
    --save_freq 10"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    5)
        print_header "Scenario 5: Quick 100-Epoch Test"
        echo "Best for: Testing configuration before long run"
        echo "Features: Shorter run with early stopping patience adjusted"
        echo ""
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 100 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 15 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 10"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    6)
        print_header "Scenario 6: Training with AutoAugment"
        echo "Best for: Maximum accuracy with advanced augmentation"
        echo "Features: AutoAugment + CutMix + label smoothing"
        echo "Note: Requires torchvision >= 0.11.0"
        echo ""
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --lr 0.1 \
    --weight_decay 5e-4 \
    --scheduler cosine \
    --early_stopping \
    --early_stopping_patience 40 \
    --autoaugment \
    --cutmix \
    --label_smoothing 0.1 \
    --save_freq 10"
        
        print_command "$CMD"
        eval $CMD
        ;;
        
    7)
        print_header "Scenario 7: Multi-Experiment Comparison"
        echo "Running 3 experiments with different configurations..."
        echo ""
        
        # Experiment 1: Baseline
        echo -e "${GREEN}Experiment 1: Baseline (no augmentation)${NC}"
        CMD1="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch_size 128 \
    --early_stopping_patience 30 \
    --csv_log ./results/exp1_baseline.csv \
    --checkpoint_dir ./checkpoints/exp1"
        echo "$CMD1"
        eval $CMD1
        
        echo ""
        
        # Experiment 2: CutMix
        echo -e "${GREEN}Experiment 2: With CutMix${NC}"
        CMD2="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch_size 128 \
    --cutmix \
    --early_stopping_patience 30 \
    --csv_log ./results/exp2_cutmix.csv \
    --checkpoint_dir ./checkpoints/exp2"
        echo "$CMD2"
        eval $CMD2
        
        echo ""
        
        # Experiment 3: CutMix + Label Smoothing
        echo -e "${GREEN}Experiment 3: CutMix + Label Smoothing${NC}"
        CMD3="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 200 \
    --batch_size 128 \
    --cutmix \
    --label_smoothing 0.1 \
    --early_stopping_patience 30 \
    --csv_log ./results/exp3_cutmix_ls.csv \
    --checkpoint_dir ./checkpoints/exp3"
        echo "$CMD3"
        eval $CMD3
        
        echo ""
        echo -e "${BLUE}Comparison complete! Check results:${NC}"
        echo "  - TensorBoard: tensorboard --logdir ./logs"
        echo "  - CSV files: ./results/exp*.csv"
        ;;
        
    *)
        echo "Invalid scenario number: $SCENARIO"
        echo "Run '$0' without arguments to see available scenarios."
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done!${NC}"
