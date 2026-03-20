## Data efficiency calculation

For a given method's validation loss, we use piecewise linear interpolation between the baseline validation loss to find the equivalent dataset size that achieves the same loss. Data efficiency is then the ratio of this equivalent size to the base token budget. For example if our method trained on 100M tokens matches the test loss of 1B tokens using a baseline, the data efficiency is 10x. 

We use nanochat as our baseline to compute data efficiency. For each token count, we train nanochat across multiple model sizes (d12, d20, d26), and take the best performance at each token count for the interpolation calculation. The validation loss for d12, d20, and d26 model sizes trained with nanochat defaults at various token counts are given here: 

|Data|d12|d20|d26|
|---|---|---|---|
|200M|3.703|3.794|3.883|
|400M|3.460|3.416|3.370|
|600M|3.356|3.270|3.200|
|800M|3.302|3.184|3.123|
|1B|3.251|3.124|3.046|
|2B|3.144|2.973|2.892|

