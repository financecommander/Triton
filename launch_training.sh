#!/bin/bash
# CIFAR-10 Training Launch Script
# 
# This script helps you launch CIFAR-10 training with the enhanced features.
# It provides different scenarios based on your needs.

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}CIFAR-10 Training Launcher${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}Python found: $(python --version)${NC}"
echo ""

# Check dependencies
echo "Checking dependencies..."
MISSING_DEPS=0

for dep in torch torchvision numpy tensorboard; do
    if python -c "import $dep" 2>/dev/null; then
        echo -e "  ✓ $dep installed"
    else
        echo -e "  ${RED}✗ $dep NOT installed${NC}"
        MISSING_DEPS=1
    fi
done

echo ""

if [ $MISSING_DEPS -eq 1 ]; then
    echo -e "${YELLOW}Missing dependencies detected!${NC}"
    echo ""
    echo "To install all required dependencies, run:"
    echo -e "  ${GREEN}pip install torch torchvision numpy tensorboard matplotlib seaborn${NC}"
    echo ""
    echo "Or install with the project:"
    echo -e "  ${GREEN}pip install -e .${NC}"
    echo ""
    read -p "Would you like to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints logs results data
echo -e "  ✓ Created: checkpoints/"
echo -e "  ✓ Created: logs/"
echo -e "  ✓ Created: results/"
echo -e "  ✓ Created: data/"
echo ""

# Check for existing checkpoints
CHECKPOINT_COUNT=$(ls checkpoints/*.pth 2>/dev/null | wc -l || echo "0")
if [ $CHECKPOINT_COUNT -gt 0 ]; then
    echo -e "${GREEN}Found $CHECKPOINT_COUNT checkpoint(s) in checkpoints/${NC}"
    echo "Available checkpoints:"
    ls -lh checkpoints/*.pth 2>/dev/null | awk '{print "  - " $9 " (" $5 ")"}'
    echo ""
fi

# Display training scenarios
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Available Training Scenarios${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "1. Fresh 500-epoch training (all enhancements)"
echo "2. Fresh 100-epoch test run (quick validation)"
echo "3. Resume from checkpoint to 500 epochs"
echo "4. Conservative training (baseline)"
echo "5. Custom command (manual input)"
echo "6. View example commands only (no execution)"
echo "7. Exit"
echo ""

read -p "Select scenario (1-7): " scenario

case $scenario in
    1)
        echo ""
        echo -e "${GREEN}Scenario 1: Fresh 500-epoch training${NC}"
        echo "Features: Early stopping, CutMix, label smoothing, TensorBoard"
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
    --csv_log ./results/fresh_500epoch.csv"
        
        echo "Command:"
        echo "$CMD"
        echo ""
        read -p "Launch training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $CMD
        fi
        ;;
        
    2)
        echo ""
        echo -e "${GREEN}Scenario 2: Quick 100-epoch test${NC}"
        echo "Features: Early stopping, CutMix, label smoothing"
        echo "Purpose: Test configuration before long run"
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
    --save_freq 10 \
    --csv_log ./results/test_100epoch.csv"
        
        echo "Command:"
        echo "$CMD"
        echo ""
        read -p "Launch training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $CMD
        fi
        ;;
        
    3)
        echo ""
        echo -e "${GREEN}Scenario 3: Resume from checkpoint${NC}"
        echo ""
        
        if [ $CHECKPOINT_COUNT -eq 0 ]; then
            echo -e "${RED}Error: No checkpoints found in checkpoints/${NC}"
            echo "Please place your checkpoint file in the checkpoints/ directory"
            exit 1
        fi
        
        echo "Available checkpoints:"
        select checkpoint in checkpoints/*.pth "Cancel"; do
            if [ "$checkpoint" = "Cancel" ]; then
                echo "Cancelled"
                exit 0
            elif [ -n "$checkpoint" ]; then
                echo ""
                echo "Selected: $checkpoint"
                break
            fi
        done
        
        CMD="python models/scripts/train_ternary_models.py \
    --model resnet18 \
    --dataset cifar10 \
    --epochs 500 \
    --batch_size 128 \
    --resume $checkpoint \
    --early_stopping \
    --early_stopping_patience 40 \
    --label_smoothing 0.1 \
    --cutmix \
    --save_freq 20 \
    --csv_log ./results/resume_to_500.csv"
        
        echo ""
        echo "Command:"
        echo "$CMD"
        echo ""
        read -p "Launch training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $CMD
        fi
        ;;
        
    4)
        echo ""
        echo -e "${GREEN}Scenario 4: Conservative baseline${NC}"
        echo "Features: Standard augmentation only, no CutMix/MixUp"
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
    --save_freq 20 \
    --csv_log ./results/conservative_500epoch.csv"
        
        echo "Command:"
        echo "$CMD"
        echo ""
        read -p "Launch training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $CMD
        fi
        ;;
        
    5)
        echo ""
        echo -e "${GREEN}Scenario 5: Custom command${NC}"
        echo ""
        echo "Enter your custom training command:"
        echo "(or press Enter for a template)"
        read -r CUSTOM_CMD
        
        if [ -z "$CUSTOM_CMD" ]; then
            CUSTOM_CMD="python models/scripts/train_ternary_models.py --model resnet18 --dataset cifar10 --epochs 100"
        fi
        
        echo ""
        echo "Command to execute:"
        echo "$CUSTOM_CMD"
        echo ""
        read -p "Launch training? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            eval $CUSTOM_CMD
        fi
        ;;
        
    6)
        echo ""
        echo -e "${BLUE}========================================${NC}"
        echo -e "${BLUE}Example Commands${NC}"
        echo -e "${BLUE}========================================${NC}"
        echo ""
        
        echo -e "${GREEN}Fresh 500-epoch training:${NC}"
        echo "python models/scripts/train_ternary_models.py \\"
        echo "    --model resnet18 --dataset cifar10 --epochs 500 \\"
        echo "    --early_stopping --early_stopping_patience 40 \\"
        echo "    --label_smoothing 0.1 --cutmix"
        echo ""
        
        echo -e "${GREEN}Resume from checkpoint:${NC}"
        echo "python models/scripts/train_ternary_models.py \\"
        echo "    --model resnet18 --dataset cifar10 --epochs 500 \\"
        echo "    --resume ./checkpoints/YOUR_CHECKPOINT.pth \\"
        echo "    --early_stopping --label_smoothing 0.1 --cutmix"
        echo ""
        
        echo -e "${GREEN}Quick 100-epoch test:${NC}"
        echo "python models/scripts/train_ternary_models.py \\"
        echo "    --model resnet18 --dataset cifar10 --epochs 100 \\"
        echo "    --early_stopping_patience 15 --cutmix"
        echo ""
        
        echo -e "${GREEN}View all options:${NC}"
        echo "python models/scripts/train_ternary_models.py --help"
        echo ""
        ;;
        
    7)
        echo "Exiting..."
        exit 0
        ;;
        
    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Training launched!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Monitor training:"
echo "  - TensorBoard: tensorboard --logdir ./logs"
echo "  - CSV logs: tail -f ./results/*.csv"
echo "  - Checkpoints: ls -lh checkpoints/"
echo ""
echo "Documentation:"
echo "  - Quick start: START_HERE.md"
echo "  - Full guide: docs/CIFAR10_TRAINING_GUIDE.md"
echo ""
