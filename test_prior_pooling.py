"""Unit test for the prior-score pooling -> expansion -> restore chain (plan Phase 5).

Guards against silent group-order misalignment between:
  * compute_group_prior_score(): score index j*groups_in + i
  * virtual_operation.forward(): p_v.view(groups_out, groups_in) then
    repeat_interleave(groups_out_dim, dim=0).repeat_interleave(groups_in_dim, dim=1)

Run standalone on CPU (no model download, no GPU):
    python test_prior_pooling.py
or via pytest.
"""

import sys

import torch

from flashlm.compression.semi_pruning_helper import (
    SemiSparseLinear,
    compute_group_prior_score,
)


def _make_module():
    # both groups_in_dim and groups_out_dim > 1 to exercise the 4-D branch
    in_dim, out_dim, groups_in_dim, groups_out_dim = 16, 12, 4, 3
    torch.manual_seed(0)
    m = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim)
    return m, in_dim, out_dim, groups_in_dim, groups_out_dim


def test_score_shape_and_normalization():
    m, in_dim, out_dim, g_in, g_out = _make_module()
    ex = m.virtual_operation.ex_dict
    G_out, G_in = ex['groups_out'], ex['groups_in']
    score = compute_group_prior_score(m.linear.weight, ex)
    assert score.numel() == m.virtual_operation.dim == G_out * G_in
    assert abs(score.mean().item()) < 1e-4, "score should be zero-mean"
    assert abs(score.std().item() - 1.0) < 1e-2, "score should be ~unit-std"


def test_marked_block_lands_on_correct_gate():
    """Mark one weight block huge -> its score must be the argmax gate, and the
    expanded mask must keep exactly that block."""
    m, in_dim, out_dim, g_in, g_out = _make_module()
    ex = m.virtual_operation.ex_dict
    G_out, G_in = ex['groups_out'], ex['groups_in']

    j_star, i_star = 2, 3  # arbitrary block within (G_out, G_in)
    with torch.no_grad():
        m.linear.weight.normal_()
        # mark the target block with a huge magnitude
        m.linear.weight[j_star * g_out:(j_star + 1) * g_out,
                        i_star * g_in:(i_star + 1) * g_in] = 1e3

    score = compute_group_prior_score(m.linear.weight, ex)
    flat_idx = int(score.argmax().item())
    assert flat_idx == j_star * G_in + i_star, (
        f"argmax gate {flat_idx} (-> block {divmod(flat_idx, G_in)}) "
        f"!= marked block ({j_star}, {i_star}): group order misaligned!"
    )

    # keep only the top-1 scored group, expand to a 2-D mask, verify placement
    with torch.no_grad():
        pv = torch.zeros_like(score)
        pv[flat_idx] = 1.0
        m.virtual_operation.set_vector_value(pv)
    m.virtual_operation.clear_cache()  # stale cached mask would mask the bug
    mask = m.virtual_operation.forward()

    assert tuple(mask.shape) == (out_dim, in_dim)
    block = mask[j_star * g_out:(j_star + 1) * g_out,
                 i_star * g_in:(i_star + 1) * g_in]
    assert bool((block == 1).all()), "marked block not fully kept"
    assert mask.sum().item() == g_out * g_in, "mask keeps more than the marked block"
    assert bool((mask[j_star * g_out:(j_star + 1) * g_out, :] == 0).sum() > 0)


def test_wanda_act_norm_shifts_argmax():
    """档位1: a large activation norm on other columns must be able to overtake
    the magnitude-only argmax, i.e. act_norm actually multiplies per input dim."""
    m, in_dim, out_dim, g_in, g_out = _make_module()
    ex = m.virtual_operation.ex_dict
    G_out, G_in = ex['groups_out'], ex['groups_in']

    score_mag = compute_group_prior_score(m.linear.weight, ex)
    act_norm = torch.ones(in_dim)
    act_norm[(G_in - 1) * g_in:] = 50.0  # boost the last in-group
    score_wanda = compute_group_prior_score(m.linear.weight, ex, act_norm=act_norm)

    assert not torch.allclose(score_mag, score_wanda)
    top_mag = divmod(int(score_mag.argmax().item()), G_in)
    top_wanda = divmod(int(score_wanda.argmax().item()), G_in)
    assert top_wanda[1] == G_in - 1, "boosted in-group should win under wanda prior"
    assert top_wanda != top_mag or top_mag[1] == G_in - 1


def test_groups_in_dim_one_branch():
    """1-D expansion branch (the production 1x256 config takes this path):
    gate k -> row k//(in_dim//R), cols [R*(k%(in_dim//R)) : +R], R = g_in_dim*g_out_dim.
    Verify pooling + expansion agree on that tile semantics."""
    in_dim, out_dim, groups_in_dim, groups_out_dim = 12, 12, 1, 3
    R = groups_in_dim * groups_out_dim          # 3
    chunks = in_dim // R                        # 4 col-chunks per row
    torch.manual_seed(1)
    m = SemiSparseLinear(in_dim, out_dim, groups_in_dim, groups_out_dim)
    ex = m.virtual_operation.ex_dict
    G_out, G_in = ex['groups_out'], ex['groups_in']

    a_star, b_star = 4, 2                       # row 4, col-chunk 2 (cols 6:9)
    with torch.no_grad():
        m.linear.weight.normal_()
        m.linear.weight[a_star, b_star * R:(b_star + 1) * R] = 1e3

    score = compute_group_prior_score(m.linear.weight, ex)
    k_star = a_star * chunks + b_star
    assert int(score.argmax().item()) == k_star, (
        f"argmax {int(score.argmax().item())} != expected gate {k_star}: "
        "1-D branch group order misaligned!"
    )

    with torch.no_grad():
        pv = torch.zeros_like(score)
        pv[k_star] = 1.0
        m.virtual_operation.set_vector_value(pv)
    m.virtual_operation.clear_cache()
    mask = m.virtual_operation.forward()
    assert bool((mask[a_star, b_star * R:(b_star + 1) * R] == 1).all()), \
        "marked tile not fully kept"
    assert mask.sum().item() == R, f"mask keeps {int(mask.sum().item())} != R={R}"


def main():
    tests = [
        test_score_shape_and_normalization,
        test_marked_block_lands_on_correct_gate,
        test_wanda_act_norm_shifts_argmax,
        test_groups_in_dim_one_branch,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"[FAIL] {t.__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(tests)} tests FAILED")
        sys.exit(1)
    print(f"\nall {len(tests)} tests passed")


if __name__ == "__main__":
    main()
