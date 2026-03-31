After some early unsuccessful attempts at full recurrence, I took a step back and asked a more basic question: if I had extra parameters to spend, should they go into width or depth? I ran matched over-budget probes in both directions and found that both had promise, with broadly comparable headline metrics.

| Probe | Change | Post-EMA val_bpb | Step avg | Steps in 1200s | Params | Size |
|---|---|---:|---:|---:|---:|---:|
| Width | `11L x 576` | `1.1277` | `~214ms` | `5,609` | `34.0M` | `19.5 MB` |
| Depth | `12L x 512` | `1.1307` | `~174ms` | `6,878` | `29.4M` | `17.3 MB` |

The results suggested that both width and depth were plausible directions. Since some of my earlier failed recurrence attempts had already explored the width side of the space, I decided to push on depth this time. That made the next question straightforward: how much of the benefit of a deeper model could I recover by reusing layers instead of paying for fully independent ones?
