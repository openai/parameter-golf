# Parameter Golf M4 Experiment Results

**Baseline**: 3.2178 BPB (71 steps)

## Results Table

| Rank | Experiment | BPB | Delta | Steps | Artifact (MB) | Time (min) |
|------|------------|-----|-------|-------|---------------|------------|
| 1 | E03_QK-Gain-4.0 | 6.7946 | +3.5768 | 30 | 5.15 | 30.8 |
| 2 | E06_Focal-Loss | 7.0688 | +3.8510 | 29 | 5.12 | 30.8 |
| 3 | E08_QK+Focal+EMA | 9.4437 | +6.2259 | 15 | 7.61 | 30.8 |
| 4 | E04_EMA-Heavy | 10.0938 | +6.8760 | 6 | 7.56 | 10.3 |
| 5 | E10_Sanath-Stack-(Partial) | 10.1494 | +6.9316 | 7 | 7.58 | 10.2 |
| 6 | E07_Stride-16-Eval | 10.1523 | +6.9345 | 7 | 5.04 | 10.1 |
| 7 | E05_Foundation-Stack | 10.1596 | +6.9418 | 6 | 7.56 | 11.3 |
| 8 | E02_Warmdown-5000 | 10.2268 | +7.0090 | 8 | 5.03 | 10.9 |
| 9 | E09_All-April-Low-Hanging | 11.0930 | +7.8752 | 8 | 7.59 | 11.2 |

## Top 3 Winners

### 1. E03_QK-Gain-4.0
- **BPB**: 6.7946 (Δ +3.5768)
- **Steps**: 30
- **Artifact**: 5.15 MB
- **Log**: /Users/sanathbs/03_Dev_Lab/projects/Personal/golf-parameter/parameter-golf/logs/exp_E03_QK-Gain-4.0.txt

### 2. E06_Focal-Loss
- **BPB**: 7.0688 (Δ +3.8510)
- **Steps**: 29
- **Artifact**: 5.12 MB
- **Log**: /Users/sanathbs/03_Dev_Lab/projects/Personal/golf-parameter/parameter-golf/logs/exp_E06_Focal-Loss.txt

### 3. E08_QK+Focal+EMA
- **BPB**: 9.4437 (Δ +6.2259)
- **Steps**: 15
- **Artifact**: 7.61 MB
- **Log**: /Users/sanathbs/03_Dev_Lab/projects/Personal/golf-parameter/parameter-golf/logs/exp_E08_QK+Focal+EMA.txt

