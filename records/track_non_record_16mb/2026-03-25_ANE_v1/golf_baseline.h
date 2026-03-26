// golf_baseline.h - GolfWide config 
#pragma once

#define MODEL_NAME "GolfWide"

#define DIM        512
#define HIDDEN     1024       // 2x DIM 
#define HEADS      8
#define KV_HEADS   4
#define HD         (DIM/HEADS)       // = 64
#define GQA_RATIO  (HEADS/KV_HEADS)  // = 2
#define Q_DIM      (HEADS * HD)      // = 512 = DIM
#define KV_DIM     (KV_HEADS * HD)   // = 256
#define SEQ        256
#define NLAYERS    9
#define VOCAB      1024

#define CKPT_PATH "ane_golf_baseline_ckpt.bin"
#define DEFAULT_DATA_PATH ".../datasets/fineweb10B_sp1024/fineweb_train_000000.bin"
