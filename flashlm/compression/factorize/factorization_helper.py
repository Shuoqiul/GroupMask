import torch


class WSVD(torch.nn.Module):
    """Placeholder base for older factorization checkpoints.

    The public cleanup does not include the original SVD replacement
    implementation. Functions below fail explicitly when that path is used.
    """


def _missing_factorize_impl(*args, **kwargs):
    raise NotImplementedError(
        "Factorization SVD helpers are not included in this public build. "
        "Use the semi-structure GroupMask path or provide a local implementation."
    )


model_replace = _missing_factorize_impl
model_SVD_and_replace = _missing_factorize_impl
get_approx_fisher = _missing_factorize_impl
Repalce_Attention = _missing_factorize_impl
hard_rank_pruning = _missing_factorize_impl
model_SVD_and_replace_test = _missing_factorize_impl


def partial_defactorize_modules(model):
    return model


def round_mid_dim(value, block_size=32):
    return max(block_size, (int(value) // block_size) * block_size)

