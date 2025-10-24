#!/bin/bash
# ------------------------------------------------------------------
# ICL Inference Batch Script
# Example usage for running ICL inference on different models
# ------------------------------------------------------------------

# Set common parameters
RESULTS_DIR="./results/icl"
MAX_SAMPLES=100  # Limit for quick testing, remove or increase for full evaluation
SEED=42

echo "Starting ICL Inference Experiments..."
echo "Results will be saved to: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR

# Function to run ICL inference for a model with different settings
run_icl_experiments() {
    local model_name=$1
    echo "Running ICL experiments for $model_name..."
    
    # Different numbers of ICL examples
    for num_examples in 0 1 3 5; do
        echo "  - Running with $num_examples examples (random selection)..."
        python src/icl_inference.py \
            --model_name $model_name \
            --num_examples $num_examples \
            --selection_method random \
            --max_samples $MAX_SAMPLES \
            --seed $SEED \
            --results_dir $RESULTS_DIR
        
        echo "    Completed!"
    done
    
    # Different selection methods (with 3 examples)
    for method in first diverse; do
        echo "  - Running with 3 examples ($method selection)..."
        python src/icl_inference.py \
            --model_name $model_name \
            --num_examples 3 \
            --selection_method $method \
            --max_samples $MAX_SAMPLES \
            --seed $SEED \
            --results_dir $RESULTS_DIR
        
        echo "    Completed!"
    done
    
    echo "Finished experiments for $model_name"
    echo ""
}

# Run experiments for all models
# Uncomment the models you want to test

# Smaller models (faster to run)
run_icl_experiments "codet5"
run_icl_experiments "plbart"

# Larger models (require more GPU memory)
# run_icl_experiments "qwen15"
# run_icl_experiments "qwen25"

echo "All ICL inference experiments completed!"
echo "Check results in: $RESULTS_DIR"
