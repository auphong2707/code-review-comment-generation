#!/usr/bin/env bash
# Eight PBS jobs; selectors group the existing internal chunk inventory.
printf '%s\n' \
    '1 smoke' \
    '2 campaign_a' \
    '3 campaign_b' \
    '4 campaign_c' \
    '5 campaign_d' \
    '6 qwen_revision' \
    '7 llama_legacy' \
    '8 llama_revision'
