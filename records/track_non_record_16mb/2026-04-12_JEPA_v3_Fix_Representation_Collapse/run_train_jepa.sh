#!/bin/bash

# Small script that activates venv, trains jepa, and closes the remote instance.
source .venv/bin/activate
python train_jepa.py
shutdown -h now