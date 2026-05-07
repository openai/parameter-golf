# FineWeb SP4096 Tokenizer

4096-vocab SentencePiece BPE tokenizer package for FineWeb.

Author: `Frosty40`

```text
⠀⠀⠀⠀⠀⠀⠀⠀
        ⠀  ⣀⠤⠚⠓⠤⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀ ⠀⠀⡾⣅⠀⠀⠀⠀⣨⢷⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⣀⡤⡦⣄⡀⠀⡇⠀⠙⢢⡔⠋⠀⢸⠀⢀⣠⢴⢤⣀⠀⠀
⡴⠊⠁⠀⢷⠀⠉⠲⣇⠀⠀⢸⡇⠀⠀⣸⠖⠉⠀⡼⠀⠈⠑⢦
⣧⠀⠀⢀⣸⡀⠀⠀⢿⠙⠲⢼⣧⠖⠋⡿⠀⠀⢀⣇⡀⠀⠀⣸
⢹⡴⠚⠉⠀⠈⠑⠦⣼⠀⠀⢸⡇⠀⠀⣧⡴⠊⠁⠀⠉⠓⢦⡏
⠀⠈⠓⢤⣀⡤⠖⠋⠁⠙⠲⣼⣧⠖⠋⠈⠙⠲⢤⣀⡤⠚⠁⠀
⢀⡠⠖⠉⠀⠉⠓⠦⣄⠴⠚⢹⡏⠓⠦⣠⡴⠚⠉⠀⠉⠲⢄⡀
⣼⠙⠲⢤⣀⡠⠔⠋⢹⠀⠀⣸⣇⠀⠀⣏⠙⠲⢄⣀⡤⠖⠋⣧
⡏⠀⠀⠀⢸⠀⠀⠀⣿⠴⠚⢹⡏⠓⠦⣿⠀ ⠀⠀⡇⠀⠀⠀⢸
⠙⠢⣄⡀⡟⣀⡤⠚⡇⠀⠀⢸⡇⠀⠀⢸⠓⠤⣀⢹⢀⣠⠔⠋
⠀⠀⠀⠉⠋⠁⠀⠀⡇⣀⠴⠊⠑⠦⣀⢸⠀⠀⠈⠙⠉⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀ ⠀⠻⣅⠀⠀⠀⠀⣨⠟⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠉⠲⠖⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
```

GitHub: `newjordan`

This folder contains the tokenizer artifacts and the scripts used to rebuild
them from the canonical repo exporter.

## Files

- `fineweb_4096_bpe.model`
- `fineweb_4096_bpe.vocab`
- `scripts/build_sp4096.py`
- `scripts/rebuild.sh`
- `scripts/tokenizer_specs.sp4096.json`
- `submission.json`

## Release Facts

- source docs sha256:
  `84386dfa7b339a5d4831d5273c4a2028b78b60670d3a235633a8520545d19bc7`
- model sha256:
  `6b0337698df13acdb58e5b05446ef26d9d7e4ca5545dc5e13ad5815a7ec904e0`
- vocab sha256:
  `925129ee4771d6f0cdb83d1284ca023cfb37360fb7b759c966cbce911ab44ad9`
- special ids:
  `pad=0`, `bos=1`, `eos=2`, `unk=3`

## Rebuild

```bash
MATCHED_FINEWEB_OUTPUT_ROOT=/tmp/sp4096_rebuild \
  bash scripts/rebuild.sh
```
