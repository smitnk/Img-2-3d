"""numpy port of the Pointcept / PTv3 point-cloud serialization UniRig's skin model uses (z-order + hilbert, plain and 'trans').

Source of truth: UniRig src/model/pointcept/models/utils/{structure.py (Point.serialization), serialization/{default,z_order,hilbert}.py}.
No torch. Integer codes are bit-exact with the torch version (tested in selftest.py against a pure-python Skilling Hilbert curve,
and in parity_x86.py against the real torch code); order = stable argsort (torch's CPU sort is stable in practice, ties = duplicate voxels).
"""
import numpy as np

ORDERS = ("z", "z-trans", "hilbert", "hilbert-trans")


def grid_coord(coord, grid_size=0.005):
    """torch.div(coord - coord.min(0), grid_size, rounding_mode='trunc').int() in float32."""
    c = np.asarray(coord, np.float32)
    return np.trunc((c - c.min(0)) / np.float32(grid_size)).astype(np.int32)


def z_encode(g, depth):
    g = np.asarray(g, np.int64)
    code = np.zeros(len(g), np.int64)
    for i in range(depth):  # x -> bit 3i+2, y -> 3i+1, z -> 3i (the LUT version is the same interleave)
        for d, off in ((0, 2), (1, 1), (2, 0)):
            code |= ((g[:, d] >> i) & 1) << (3 * i + off)
    return code


def hilbert_encode(g, depth):
    """Skilling's transpose algorithm exactly as written in serialization/hilbert.py (bits kept MSB first, one uint8 per bit)."""
    g = np.asarray(g, np.int64)
    n, B = len(g), depth
    gray = ((g[:, :, None] >> np.arange(B - 1, -1, -1)) & 1).astype(np.uint8)  # (n, 3, B)
    for bit in range(B):
        for dim in range(3):
            mask = gray[:, dim, bit].copy()
            gray[:, 0, bit + 1:] ^= mask[:, None]
            flip = (1 - mask)[:, None] & (gray[:, 0, bit + 1:] ^ gray[:, dim, bit + 1:])
            gray[:, dim, bit + 1:] ^= flip
            gray[:, 0, bit + 1:] ^= flip
    bits = gray.swapaxes(1, 2).reshape(n, B * 3)  # bit-major, then dim
    L = bits.shape[1]
    shift = 1 << (int(np.ceil(np.log2(L))) - 1)  # gray2binary: prefix xor by doubling shifts
    while shift > 0:
        if shift < L:
            old = bits.copy()
            bits[:, shift:] ^= old[:, :L - shift]
        shift //= 2
    return (bits.astype(np.int64) << np.arange(L - 1, -1, -1)).sum(1)


def encode(g, depth, order):
    assert order in ORDERS
    g = np.asarray(g)
    if order.endswith("-trans"):
        g = g[:, [1, 0, 2]]
    return (z_encode if order.startswith("z") else hilbert_encode)(g, depth)


def serialize(coord, grid_size=0.005, orders=ORDERS):
    """Point.serialization(order=orders, shuffle_orders=False) for one cloud -> dict(grid (N,3), code/order/inverse (len(orders), N))."""
    g = grid_coord(coord, grid_size)
    depth = int(g.max()).bit_length()
    assert depth * 3 <= 63 and depth <= 16
    code = np.stack([encode(g, depth, o) for o in orders])
    order = np.argsort(code, axis=1, kind="stable")
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.broadcast_to(np.arange(code.shape[1]), order.shape), 1)
    return {"grid": g, "code": code, "order": order, "inverse": inverse}
