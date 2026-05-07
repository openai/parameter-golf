# AR vs Mercury Decode Benchmark

## Setup
- Baseline checkpoint: `/workspace/parameter-golf/records/track_non_record_16mb/2026-04-09_MercuryStyleCompactTextDiffusion/benchmarks/real_8x_seed2026_vs_baseline_20260410_140157/ar_baseline/final_model.pt`
- Mercury checkpoint: `/workspace/parameter-golf/records/track_non_record_16mb/2026-04-09_MercuryStyleCompactTextDiffusion/final_model.pt`
- Device: `cuda`
- Examples: `32`
- Continuation task: prefix `128` tokens, predict next `64` tokens
- Infill task: prefix `128` tokens, infill `64` tokens with suffix `64` visible

## Highlights

- AR continuation throughput: `1518.79` tok/s for `64`-token greedy decode.
- Best Mercury continuation accuracy in this run: `0.0400` at `1` refinement step(s).
- Fastest Mercury continuation setting: `52423.93` tok/s at `2` step(s), which is `34.52x` AR continuation throughput.
- Fastest Mercury infill setting: `147729.24` tok/s at `1` step(s), which is `97.27x` AR continuation throughput on the same hardware.

## Continuation

| Model | Mode | Token Acc | Exact Seq | Tok/s | Speedup vs AR | Batch Latency ms | Single-example Latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AR baseline | greedy | 0.0215 | 0.0000 | 1518.79 | 1.00x | 1348.44 | 510.46 |
| Mercury | 1 refinement step(s) | 0.0400 | 0.0000 | 33315.36 | 21.94x | 61.47 | 7.68 |
| Mercury | 2 refinement step(s) | 0.0400 | 0.0000 | 52423.93 | 34.52x | 39.07 | 15.34 |
| Mercury | 4 refinement step(s) | 0.0396 | 0.0000 | 43382.67 | 28.56x | 47.21 | 30.57 |
| Mercury | 8 refinement step(s) | 0.0376 | 0.0000 | 21693.85 | 14.28x | 94.40 | 61.32 |

## Infill

| Model | Mode | Token Acc | Exact Seq | Tok/s | Speedup vs AR Continuation | Batch Latency ms | Single-example Latency ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mercury | 1 refinement step(s) | 0.0400 | 0.0000 | 147729.24 | 97.27x | 13.86 | 7.63 |
| Mercury | 2 refinement step(s) | 0.0400 | 0.0000 | 86371.16 | 56.87x | 23.71 | 15.29 |
| Mercury | 4 refinement step(s) | 0.0396 | 0.0000 | 43263.39 | 28.49x | 47.34 | 30.28 |
| Mercury | 8 refinement step(s) | 0.0376 | 0.0000 | 21754.90 | 14.32x | 94.14 | 44.70 |

## Example Outputs

### Example 1
- Prefix: customers and building brand awareness through social media initiatives *Performing website analyses utilizing Google Analytics *Participating in planning, executing and analyzing email marketing campaigns - Web Design *Designing web imagery used on Mason Companies websites, display advert
- Continuation target: ising and landing pages *Designing imagery and HTML for email marketing campaigns *Participating in various projects such as launching new mobile sit
- AR continuation: ising, and marketing - Web Design *Producing web design ideas for websites, websites, and websites - Web Design *Producing web design ideas for we
- Mercury continuation (1 step): isment,,,,,,,,,,,,,,inginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginging,inginging
- Infill target: ising and landing pages *Designing imagery and HTML for email marketing campaigns *Participating in various projects such as launching new mobile sit
- Infill suffix: es and websites *Researching and implementing emerging web design technologies and processes - IT Developer *Java J2EE development *Ecommerce Development on Mason
- Mercury infill (1 step): isment,,,,,,,,,,,,,,inginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginginging,inginging

### Example 2
- Prefix: and are responsible for compliance with all applicable laws. You may not access, download, use or export the Information on this web site in violation of U.S. export laws or regulations, or in violation of any applicable local laws or regulations. Webroot Inc. ("Webroot") is committed to protecting the intellectual property rights of third parties, and
- Continuation target: Webroot requests that its customers and community members do the same. Webroot has no responsibility for content on other websites that you may find or access when using Webroot's produ
- AR continuation: to protecting the privacy of third parties. Webroot is not responsible for the privacy of third parties. Webroot is not responsible for the privacy of third part
- Mercury continuation (1 step): theinging the the,,,ss,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
- Infill target: Webroot requests that its customers and community members do the same. Webroot has no responsibility for content on other websites that you may find or access when using Webroot's produ
- Infill suffix: cts or services, and such content may be protected by copyright and the intellectual property laws of the United States and/or other countries. Without prior notice and at any time, We
- Mercury infill (1 step): theinging the the,,,ss,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,

### Example 3
- Prefix: os predicted will emerge without deliberate and intentional actions to support them. And the extent to which they can be shaped to further societal goals will depend on constructive dialogue between governments and citizens themselves. Ultimately, this new publication aims to contribute to this dialogue, so that both developing and developed countries
- Continuation target: are more likely to leap into better futures. Text co-authored with Tom Steinberg, originally cross-posted from the World Bank’s Governance for Development
- AR continuation: can be more effectively dialogued and delivered to the public. The publication is aimed at the public and private sectors, and is based on the publication’s publication d
- Mercury continuation (1 step): will the the to the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
- Infill target: are more likely to leap into better futures. Text co-authored with Tom Steinberg, originally cross-posted from the World Bank’s Governance for Development
- Infill suffix: blog. You can also read another article about this report in Apolitical here. While I’m at it: if you work in public service and care about making government work better, I highly recommend
- Mercury infill (1 step): will the the to the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the the
