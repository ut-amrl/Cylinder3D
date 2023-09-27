#!/bin/bash

default_gpuid=0

if [ $# -gt 0 ]; then
    # if an argumenet is provided, it sets that to the GPU number
    CUDA_VISIBLE_DEVICES=$1
else
    CUDA_VISIBLE_DEVICES=${default_gpuid}
fi