#!/bin/bash

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=2 \
    -m pcg.pcg_exec_tests.2LayerMLP

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node=1 \
    -m pcg.pcg_exec_tests.linear_2LayerMLP

python compare_files.py