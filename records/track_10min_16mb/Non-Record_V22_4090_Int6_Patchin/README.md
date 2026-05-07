# Non-Record: V22 Architecture on Single RTX 4090 (val_bpb=1.4537)

**Team:** Patchin Development Room RC-zo

We are submitting a non-record run to demonstrate our V22 architecture (featuring Muon optimizer tuning and INT6 quantization). Due to budget constraints, this was trained on a single RTX 4090 spot instance rather than the official 8xH100 setup.

## Results
- **Hardware:** 1x RTX 4090
- **Training Time:** ~43.5 minutes (2613s) / 5000 steps
- **Artifact Size:** 11.38 MB
- **val_bpb:** 1.4537 (sampled at step 5000)

## Key Techniques
- **Optimizer:** Highly tuned Muon optimizer with specific scheduling.
- **Quantization:** INT6 quantization combined with zlib compression (level 9) to aggressively crush the artifact size down to 11.38 MB.
- **Path/Execution Stability:** Fully hardcoded absolute paths to prevent any I/O bottlenecks or environment variable desyncs during single-node DDP runs.

## Training Log Snippet
```text
🚀 1/4: 必要なライブラリをインストールしています...
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager, possibly rendering your system unusable.It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv. Use the --root-user-action option if you know what you are doing and want to suppress this warning.

[notice] A new release of pip is available: 24.2 -> 26.0.1
[notice] To update, run: python -m pip install --upgrade pip
🚀 2/4: ゴミを掃除し、公式リポジトリを確実に取得しています...
Cloning into 'parameter-golf'...
Updating files: 100% (227/227), done.
🚀 3/4: 学習データと「辞書」を【正しい公式コマンドで】取得しています...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
✅ データの準備が完璧に整いました！
🚀 4/4: V22（5000歩完走・完全修正版）のコードをセットしました！
🔥 いざ発射！5000歩完走特化・トレーニング開始！ 🔥
Val tokens: 62021846
Params: 16,892,544
Training start (Target: 5000 iterations)
step      1/5000 loss 0.7019 lr 1.50e-05 time 1s
step     50/5000 loss 6.0525 lr 7.50e-04 time 26s
step    100/5000 loss 5.7919 lr 1.50e-03 time 51s
step    150/5000 loss 4.5283 lr 2.25e-03 time 77s
step    200/5000 loss 4.3465 lr 3.00e-03 time 102s
step    250/5000 loss 4.2668 lr 3.00e-03 time 127s
step    300/5000 loss 4.1990 lr 3.00e-03 time 152s
step    350/5000 loss 4.1932 lr 2.99e-03 time 178s
step    400/5000 loss 4.1236 lr 2.99e-03 time 203s
step    450/5000 loss 3.9401 lr 2.98e-03 time 228s
step    500/5000 loss 3.6975 lr 2.97e-03 time 254s
step    550/5000 loss 3.5536 lr 2.96e-03 time 279s
step    600/5000 loss 3.4314 lr 2.95e-03 time 304s
step    650/5000 loss 3.3318 lr 2.94e-03 time 329s
step    700/5000 loss 3.2414 lr 2.92e-03 time 354s
step    750/5000 loss 3.2350 lr 2.91e-03 time 380s
step    800/5000 loss 3.1729 lr 2.89e-03 time 405s
step    850/5000 loss 3.1476 lr 2.87e-03 time 430s
step    900/5000 loss 3.0876 lr 2.85e-03 time 455s
step    950/5000 loss 3.0429 lr 2.83e-03 time 480s
step   1000/5000 loss 2.9944 lr 2.81e-03 time 506s
--- 途中評価開始（時間節約のためサンプリング） ---
--- step 1000 | val_loss 4.5304 | val_bpb 2.7154
step   1050/5000 loss 2.9891 lr 2.78e-03 time 552s
step   1100/5000 loss 2.8761 lr 2.76e-03 time 578s
step   1150/5000 loss 2.8751 lr 2.73e-03 time 603s
step   1200/5000 loss 2.8626 lr 2.70e-03 time 628s
step   1250/5000 loss 2.8124 lr 2.67e-03 time 653s
step   1300/5000 loss 2.7451 lr 2.64e-03 time 678s
step   1350/5000 loss 2.7609 lr 2.61e-03 time 704s
step   1400/5000 loss 2.7183 lr 2.58e-03 time 729s
step   1450/5000 loss 2.7212 lr 2.54e-03 time 754s
step   1500/5000 loss 2.7198 lr 2.51e-03 time 779s
step   1550/5000 loss 2.6885 lr 2.47e-03 time 812s
step   1600/5000 loss 2.6385 lr 2.43e-03 time 837s
step   1650/5000 loss 2.6314 lr 2.39e-03 time 862s
step   1700/5000 loss 2.6807 lr 2.36e-03 time 887s
step   1750/5000 loss 2.6470 lr 2.32e-03 time 912s
step   1800/5000 loss 2.6605 lr 2.28e-03 time 937s
step   1850/5000 loss 2.6430 lr 2.23e-03 time 962s
step   1900/5000 loss 2.6091 lr 2.19e-03 time 987s
step   1950/5000 loss 2.6013 lr 2.15e-03 time 1012s
step   2000/5000 loss 2.6062 lr 2.10e-03 time 1037s
--- 途中評価開始（時間節約のためサンプリング） ---
--- step 2000 | val_loss 3.0974 | val_bpb 1.8565
step   2050/5000 loss 2.5944 lr 2.06e-03 time 1080s
step   2100/5000 loss 2.6072 lr 2.02e-03 time 1105s
step   2150/5000 loss 2.5986 lr 1.97e-03 time 1130s
step   2200/5000 loss 2.5692 lr 1.93e-03 time 1155s
step   2250/5000 loss 2.5309 lr 1.88e-03 time 1180s
step   2300/5000 loss 2.5481 lr 1.83e-03 time 1205s
step   2350/5000 loss 2.5839 lr 1.79e-03 time 1230s
step   2400/5000 loss 2.5356 lr 1.74e-03 time 1255s
step   2450/5000 loss 2.5528 lr 1.69e-03 time 1280s
step   2500/5000 loss 2.5586 lr 1.64e-03 time 1305s
step   2550/5000 loss 2.5082 lr 1.60e-03 time 1330s
step   2600/5000 loss 2.5512 lr 1.55e-03 time 1355s
step   2650/5000 loss 2.5246 lr 1.50e-03 time 1380s
step   2700/5000 loss 2.5179 lr 1.46e-03 time 1405s
step   2750/5000 loss 2.5273 lr 1.41e-03 time 1430s
step   2800/5000 loss 2.4948 lr 1.36e-03 time 1455s
step   2850/5000 loss 2.5443 lr 1.31e-03 time 1480s
step   2900/5000 loss 2.4970 lr 1.27e-03 time 1505s
step   2950/5000 loss 2.5002 lr 1.22e-03 time 1530s
step   3000/5000 loss 2.5256 lr 1.17e-03 time 1555s
--- 途中評価開始（時間節約のためサンプリング） ---
--- step 3000 | val_loss 2.5954 | val_bpb 1.5556
step   3050/5000 loss 2.5238 lr 1.13e-03 time 1601s
step   3100/5000 loss 2.5685 lr 1.08e-03 time 1634s
step   3150/5000 loss 2.4766 lr 1.04e-03 time 1659s
step   3200/5000 loss 2.3647 lr 9.95e-04 time 1684s
step   3250/5000 loss 2.4868 lr 9.52e-04 time 1709s
step   3300/5000 loss 2.5053 lr 9.09e-04 time 1735s
step   3350/5000 loss 2.4785 lr 8.66e-04 time 1760s
step   3400/5000 loss 2.4508 lr 8.25e-04 time 1785s
step   3450/5000 loss 2.4594 lr 7.84e-04 time 1810s
step   3500/5000 loss 2.4202 lr 7.44e-04 time 1835s
step   3550/5000 loss 2.4791 lr 7.05e-04 time 1861s
step   3600/5000 loss 2.4482 lr 6.67e-04 time 1886s
step   3650/5000 loss 2.3596 lr 6.30e-04 time 1911s
step   3700/5000 loss 2.3958 lr 5.94e-04 time 1936s
step   3750/5000 loss 2.4689 lr 5.59e-04 time 1962s
step   3800/5000 loss 2.4529 lr 5.25e-04 time 1987s
step   3850/5000 loss 2.4463 lr 4.92e-04 time 2012s
step   3900/5000 loss 2.4166 lr 4.60e-04 time 2037s
step   3950/5000 loss 2.4664 lr 4.29e-04 time 2062s
step   4000/5000 loss 2.4661 lr 4.00e-04 time 2088s
--- 途中評価開始（時間節約のためサンプリング） ---
--- step 4000 | val_loss 2.4659 | val_bpb 1.4780
step   4050/5000 loss 2.4509 lr 3.71e-04 time 2135s
step   4100/5000 loss 2.3623 lr 3.44e-04 time 2160s
step   4150/5000 loss 2.4386 lr 3.19e-04 time 2185s
step   4200/5000 loss 2.5189 lr 2.94e-04 time 2210s
step   4250/5000 loss 2.4222 lr 2.71e-04 time 2235s
step   4300/5000 loss 2.4151 lr 2.50e-04 time 2260s
step   4350/5000 loss 2.4037 lr 2.29e-04 time 2286s
step   4400/5000 loss 2.3919 lr 2.10e-04 time 2311s
step   4450/5000 loss 2.4449 lr 1.93e-04 time 2336s
step   4500/5000 loss 2.4293 lr 1.77e-04 time 2361s
step   4550/5000 loss 2.3579 lr 1.62e-04 time 2386s
step   4600/5000 loss 2.4019 lr 1.49e-04 time 2412s
step   4650/5000 loss 2.4737 lr 1.38e-04 time 2437s
step   4700/5000 loss 2.4142 lr 1.28e-04 time 2462s
step   4750/5000 loss 2.3557 lr 1.19e-04 time 2488s
step   4800/5000 loss 2.4007 lr 1.12e-04 time 2513s
step   4850/5000 loss 2.4598 lr 1.07e-04 time 2538s
step   4900/5000 loss 2.4407 lr 1.03e-04 time 2563s
step   4950/5000 loss 2.4468 lr 1.01e-04 time 2588s
step   5000/5000 loss 2.4312 lr 1.00e-04 time 2613s
--- 途中評価開始（時間節約のためサンプリング） ---
--- step 5000 | val_loss 2.4253 | val_bpb 1.4537

--- 最終評価（フルスキャン）開始 ---
