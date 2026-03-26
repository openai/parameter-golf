#   # Exp 1: Bigger batch (isolate gradient quality)
#   ./run_experiment.sh exp1_batch_65k ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=50

#   # Exp 2: Layer sharing iso-depth (3 unique × 3 repeats = 9 effective)
#   ./run_experiment.sh exp2_share_3x3 TRAIN_SCRIPT=train_gpt_exp.py ITERATIONS=300 TRAIN_BATCH_TOKENS=16384 NUM_LAYERS=9 NUM_UNIQUE_LAYERS=3 VAL_LOSS_EVERY=50

#   # Exp 3: Deep sharing (3 unique × 6 repeats = 18 effective)
#   ./run_experiment.sh exp3_share_3x6 TRAIN_SCRIPT=train_gpt_exp.py ITERATIONS=300 TRAIN_BATCH_TOKENS=16384 NUM_LAYERS=18 NUM_UNIQUE_LAYERS=3 VAL_LOSS_EVERY=50

#   # Exp 4: Deep-thin no sharing (18 layers, 384 dim)
#   ./run_experiment.sh exp4_deep_thin ITERATIONS=300 TRAIN_BATCH_TOKENS=16384 NUM_LAYERS=18 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 VAL_LOSS_EVERY=50

#   # Exp 5: Combined (6 unique × 3 repeats = 18 eff, 512 dim, big batch)
#   ./run_experiment.sh exp5_combined TRAIN_SCRIPT=train_gpt_exp.py ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 NUM_LAYERS=18 NUM_UNIQUE_LAYERS=6 MODEL_DIM=512 VAL_LOSS_EVERY=50


# Exp 6: Same as exp1 but more iterations (see if 65K batch keeps improving)
./run_experiment.sh exp6_batch_65k_1000 ITERATIONS=1000 TRAIN_BATCH_TOKENS=65536 VAL_LOSS_EVERY=200

# Exp 7: 6 unique layers × 2 repeats = 12 eff, 512 dim, 65K batch
# ~12M params, should fit in 16MB, gets both depth and bigger batch
./run_experiment.sh exp7_share_6x2_65k TRAIN_SCRIPT=train_gpt_exp.py ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 NUM_LAYERS=12 NUM_UNIQUE_LAYERS=6 MODEL_DIM=512 VAL_LOSS_EVERY=50

# Exp 8: 9 unique layers × 2 repeats = 18 eff, 512 dim, 65K batch
# Full 17M params + double depth via sharing
./run_experiment.sh exp8_share_9x2_65k TRAIN_SCRIPT=train_gpt_exp.py ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 NUM_LAYERS=18 NUM_UNIQUE_LAYERS=9 MODEL_DIM=512 VAL_LOSS_EVERY=50

# Exp 9: Wider model - 9 layers, 640 dim, 65K batch (fills more of 16MB budget)
./run_experiment.sh exp9_wider_640 ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 NUM_LAYERS=9 MODEL_DIM=640 NUM_HEADS=10 NUM_KV_HEADS=5 VAL_LOSS_EVERY=50

# Exp 10: Deep-thin + big batch (18 layers, 384 dim, 65K batch)
./run_experiment.sh exp10_deep_thin_65k ITERATIONS=300 TRAIN_BATCH_TOKENS=65536 NUM_LAYERS=18 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=3 VAL_LOSS_EVERY=50