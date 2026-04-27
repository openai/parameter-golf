#!/usr/bin/env python3

# Keep the top-level script intentionally tiny: the real implementation lives in
# the library package so it can be imported by tests, reused from other tools,
# and kept separate from CLI-entry boilerplate. That separation also makes it
# easier to evolve the training loop without accumulating more "if __name__ ==
# '__main__'" concerns in the same file.
from train_gpt_mlx_lib.runner import main


if __name__ == "__main__":
    main()
