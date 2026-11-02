"""CPU stand-in for flash-attention 3's flash_attn_func (SkinTokens imports it unconditionally).
Same layout: q, k, v are (batch, seq, heads, dim); returns (out, None) like FA3."""
import torch.nn.functional as F


def flash_attn_func(q, k, v, *args, causal=False, **kwargs):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal)
    return out.transpose(1, 2), None
