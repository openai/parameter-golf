# 進捗まとめ (2026-04-09)

## 環境クリーンアップ (RunPod)
- コンテナディスク `/` が 16G/20G (77%) で逼迫していた
- 原因: `/root/.cache/huggingface` が 16G (parameter-golf データセットの重複DL)
- 対処: HFキャッシュを削除 (データ本体は `/workspace/parameter-golf/data` に既存)
- 結果: `/` = **695M / 20G (4%)** に回復

## ベースライン実行結果 (`logs/baseline_1800s.log`) — RTX 4090

### 設定
- attention: GQA (heads=8, kv_heads=4), flash SDPA
- tie_embeddings: True
- LR: embed=0.05, matrix=0.04, scalar=0.04, head=0.0
- batch_tokens=524288, seq_len=1024, iters=20000 (cap)
- wallclock cap: **1800s**
- seed: 1337

### 学習
- 早期停止: wallclock cap で **step 1942 / 20000** で停止
- step time: ~927 ms/step
- step 0   : val_loss=6.9357 / val_bpb=4.1077
- step 1942: val_loss=**2.1914** / val_bpb=**1.2979**
- peak mem: 10829 MiB allocated / 11374 MiB reserved

### val_bpb 推移
| step | val_bpb |
|------|---------|
| 0    | 4.1077  |
| 200  | 1.6837  |
| 400  | 1.5199  |
| 600  | 1.4505  |
| 800  | 1.4084  |
| 1000 | 1.3736  |
| 1200 | 1.3498  |
| 1400 | 1.3319  |
| 1600 | 1.3174  |
| 1800 | 1.3041  |
| 1942 | **1.2979** |

### 提出物サイズ
| 項目 | サイズ |
|---|---|
| Serialized model | 67,224,983 B |
| Code | 47,880 B |
| **Total submission** | **67,272,863 B (~64 MiB)** |
| Serialized model int8+zlib | 14,881,575 B (3.91x圧縮) |
| **Total submission int8+zlib** | **14,929,455 B (~14.2 MiB)** |

### int8+zlib ラウンドトリップ評価
- val_loss = **2.19363408**
- val_bpb  = **1.29919366**
- 量子化劣化: bpb +0.0013 (1.2979 → 1.2992)
- eval_time: 32,679 ms

## exp1: SmearGate + Sliding Window Eval — RTX 4090

### 変更点
- SmearGate アーキテクチャを適用
- 評価モードを `sliding_window` (stride=64, batch_seqs=32) に変更

### 学習
- 設定はベースラインと同じ (GQA 8h/4kv, tie_emb, 同LR, seq_len=1024, 1800s cap)
- 早期停止: step **1940 / 20000** (wallclock cap)
- step time: ~928 ms/step
- step 1940: val_bpb=**1.2969**

### val_bpb 推移 (学習中評価)
| step | val_bpb |
|------|---------|
| 0    | 4.1075  |
| 200  | 1.6848  |
| 400  | 1.5149  |
| 600  | 1.4507  |
| 800  | 1.4061  |
| 1000 | 1.3726  |
| 1200 | 1.3484  |
| 1400 | 1.3305  |
| 1600 | 1.3164  |
| 1800 | 1.3030  |
| 1940 | **1.2969** |

### 提出物サイズ
| 項目 | サイズ |
|---|---|
| Serialized model | 67,281,771 B |
| Serialized model int8+zlib | 14,873,087 B (3.91x圧縮) |
| **Total submission int8+zlib** | **14,927,442 B (~14.2 MiB)** |

### Sliding Window 最終評価 (int8+zlib roundtrip)
- val_loss = **2.13559351**
- val_bpb  = **1.26482039**
- peak mem: 10892 MiB allocated / 11238 MiB reserved
- eval_time: 1,451,742 ms (~24分)

### ベースライン比較
| 実験 | 評価方法 | val_bpb (int8+zlib) |
|------|---------|---------------------|
| baseline | 通常eval | 1.2992 |
| **exp1 SmearGate+SW** | **sliding_window stride=64** | **1.2648** |
| 改善 | | **-0.0344** |

> sliding window 評価によって bpb が 1.2992 → **1.2648** に改善 (記録更新)

## 次にやること候補
1. SmearGate+SW をベースに追加改善実験 (現在 bpb **1.2648**)
2. 既存記録 `val_bpb 1.11473` (AR Self-Gen GPTQ + XSA + BigramHash) との差分分析
