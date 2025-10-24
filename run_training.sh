#!/bin/bash

# Script to run training with proper paths from project root
# Usage:
#   SWEEP: ./run_training.sh <model> <method> sweep <start_exp> <end_exp> [subset=1]
#   FIXED: ./run_training.sh <model> <method> fixed <learning_rate> [subset=1]
#
# Examples:
#   ./run_training.sh qwen15 ia3 sweep 0 4 0.2
#   ./run_training.sh qwen15 ia3 fixed 5e-5

MODEL=$1
METHOD=$2
MODE=$3  # may be "sweep" | "fixed" 

if [ -z "$MODEL" ] || [ -z "$METHOD" ]; then
  echo "Usage:"
  echo "  SWEEP: $0 <model> <method> sweep <start_exp> <end_exp> [subset=1]"
  echo "  FIXED: $0 <model> <method> fixed <learning_rate> [subset=1]"
  echo "  Legacy auto-detect also supported."
  echo "Models: codet5, plbart, qwen15, qwen25, qwen25-0.5b"
  echo "Methods: fmft, ia3, lora, qlora"
  exit 1
fi

# Helpers
is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
is_number()  { [[ "$1" =~ ^[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$ ]]; }

# Defaults
SUBSET_DEFAULT=1

# Set Python path to include src directory and current directory
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/src"

case "$MODEL" in
  "codet5") MODEL_DIR="codet5"; SCRIPT_PREFIX="ct5" ;;
  "plbart") MODEL_DIR="plbart"; SCRIPT_PREFIX="plbart" ;;
  "qwen15") MODEL_DIR="qwen15"; SCRIPT_PREFIX="qwen15" ;;
  "qwen25") MODEL_DIR="qwen25"; SCRIPT_PREFIX="qwen25" ;;
  "qwen25-0.5b") MODEL_DIR="qwen25-0.5b-instruct"; SCRIPT_PREFIX="qwen25_0.5b" ;;
  *) echo "Invalid model: $MODEL"; exit 1 ;;
esac

SCRIPT_PATH="src/models/$MODEL_DIR/main_${SCRIPT_PREFIX}_${METHOD}.py"
if [ ! -f "$SCRIPT_PATH" ]; then
  echo "Script not found: $SCRIPT_PATH"
  exit 1
fi

run_single() {
  local subset_val="$1"
  local lr_val="$2"   # may be empty (None)
  echo "-----------------------------------------"
  echo "Running single: subset=${subset_val}, lr=${lr_val:-default}"
  CMD="python \"$SCRIPT_PATH\" --subset \"$subset_val\""
  if [ -n "$lr_val" ]; then
    CMD="$CMD --learning_rate \"$lr_val\""
  fi
  echo "CMD: $CMD"
  eval $CMD
  return $?
}

run_experiment() {
  local exp_num="$1"
  local subset_val="$2"

  echo "========================================="
  echo "Experiment $exp_num | Model: $MODEL | Method: $METHOD"
  echo "Subset: ${subset_val} | LR: (handled in Python sweep)"
  echo "========================================="

  CMD="python \"$SCRIPT_PATH\" --experiment_id \"$exp_num\" --subset \"$subset_val\""

  echo "CMD: $CMD"
  eval $CMD
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    echo "ERROR: Experiment $exp_num failed with exit code $exit_code"
    return $exit_code
  fi
  echo "Experiment $exp_num completed successfully"
  echo ""
}


# Parse modes and args
if [ "$MODE" = "sweep" ]; then
    START_EXP=$4
    END_EXP=$5
    SUBSET=${6:-$SUBSET_DEFAULT}

    if ! is_integer "$START_EXP" || ! is_integer "$END_EXP"; then
        echo "Error: start_exp and end_exp must be integers"
        exit 1
    fi
    if (( START_EXP > END_EXP )); then
        echo "Error: start_exp ($START_EXP) cannot be greater than end_exp ($END_EXP)"
        exit 1
    fi
    # Validate subset is number in [0,1]
    if ! is_number "$SUBSET" || (( $(echo "$SUBSET < 0 || $SUBSET > 1" | bc -l) )); then
        echo "Error: subset must be a number between 0 and 1"
        exit 1
    fi

    echo "Running sweep from $START_EXP to $END_EXP on subset=$SUBSET"
    for i in $(seq "$START_EXP" "$END_EXP"); do
        run_experiment "$i" "$SUBSET" ""  # no LR passed → Python handles LR search
        if [ $? -ne 0 ]; then
            echo "Stopping due to failed experiment $i"
            exit 1
        fi
    done
    echo "All experiments ($START_EXP..$END_EXP) completed!"

elif [ "$MODE" = "fixed" ]; then
    LR_VAL=$4
    SUBSET=${5:-$SUBSET_DEFAULT}
    if ! is_number "$LR_VAL"; then
        echo "Error: learning_rate must be a number (e.g., 5e-5, 0.00005)"
        exit 1
    fi
    if ! is_number "$SUBSET" || (( $(echo "$SUBSET < 0 || $SUBSET > 1" | bc -l) )); then
        echo "Error: subset must be a number between 0 and 1"
        exit 1
    fi

    # Single run (no experiment_id provided → python default behavior)
    run_single "$SUBSET" "$LR_VAL" || exit $?
fi
