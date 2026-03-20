Best raw checkpoint metadata captured from the Runpod pod before shutdown:

- Run ID: `twice_eval2048_ttt1024`
- Remote path: `/workspace/parameter-golf/final_model.pt`
- Size on pod: `72M`
- SHA256: `292d79fa54a638be348354f09d185f80b69710e7de8f4dfa42b36e43afccdc96`

The raw `.pt` file itself was not copied into this repo because Runpod's SSH wrapper blocked automated binary transfer through `scp`. If you want to preserve the raw checkpoint, keep the pod or its volume alive until we manually copy it out tomorrow.
