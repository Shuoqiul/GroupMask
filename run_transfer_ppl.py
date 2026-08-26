"""Compatibility entrypoint for transfer PPL evaluation.

The implementation lives in hf_ppl.py. This wrapper keeps older commands that
call run_transfer_ppl.py working.
"""

import torch
from jsonargparse import CLI

from hf_ppl import main


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    CLI(main)
