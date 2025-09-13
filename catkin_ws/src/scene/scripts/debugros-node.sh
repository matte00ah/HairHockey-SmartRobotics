#!/bin/bash
source $(conda info --base)/etc/profile.d/conda.sh

# Activate your Conda environment
conda activate test_env

# Run the ROS node
rosrun scene prediction.py
 