"""numpy port of the Pointcept / PTv3 point-cloud serialization UniRig's skin model uses (z-order + hilbert, plain and 'trans').

Source of truth: UniRig src/model/pointcept/models/utils/{structure.py (Point.serialization), serialization/{default,z_order,hilbert}.py}.
No torch. Integer codes are bit-exact with the torch version (tested in selftest.py against a pure-python Skilling Hilbert curve,
and in parity_x86.py against the real torch code). order = torch.argsort(code) (stable=False): on CPU that is libstdc++'s std::sort, whose
order of equal codes (duplicate voxels) is deterministic but not stable, so it is emulated (std_argsort); tie='stable' is the plain stable sort.
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


def std_argsort(keys):
    """Indices that libstdc++ std::sort (introsort, key-only comparator, move_median_to_first pivot) gives on (key, index) pairs.
    shortcut: depth-limit exhaustion (heapsort fallback, never seen on real clouds) falls back to a stable sort of that range."""
    k = [int(x) for x in keys]
    ix = list(range(len(k)))
    n = len(k)

    def swap(a, b):
        k[a], k[b] = k[b], k[a]
        ix[a], ix[b] = ix[b], ix[a]

    def loop(lo, hi, depth):
        while hi - lo > 16:
            if depth == 0:
                seg = sorted(range(lo, hi), key=lambda i: k[i])
                kk, ii = [k[i] for i in seg], [ix[i] for i in seg]
                k[lo:hi], ix[lo:hi] = kk, ii
                return
            depth -= 1
            mid = lo + (hi - lo) // 2
            a, b, c = lo + 1, mid, hi - 1  # move_median_to_first(result=lo, a, b, c)
            if k[a] < k[b]:
                swap(lo, b if k[b] < k[c] else (c if k[a] < k[c] else a))
            else:
                swap(lo, a if k[a] < k[c] else (c if k[b] < k[c] else b))
            pv = k[lo]
            f, l = lo + 1, hi
            while True:  # unguarded_partition(first=lo+1, last=hi, pivot=lo)
                while k[f] < pv:
                    f += 1
                l -= 1
                while pv < k[l]:
                    l -= 1
                if not f < l:
                    break
                swap(f, l)
                f += 1
            loop(f, hi, depth)
            hi = f

    def ins(i):  # unguarded_linear_insert
        v, vi, j = k[i], ix[i], i
        while v < k[j - 1]:
            k[j], ix[j] = k[j - 1], ix[j - 1]
            j -= 1
        k[j], ix[j] = v, vi

    if n > 1:
        loop(0, n, 2 * (n.bit_length() - 1))
        top = min(n, 16)
        for i in range(1, top):  # insertion_sort(first, first+16)
            if k[i] < k[0]:
                v, vi = k[i], ix[i]
                k[1:i + 1], ix[1:i + 1] = k[0:i], ix[0:i]
                k[0], ix[0] = v, vi
            else:
                ins(i)
        for i in range(top, n):  # unguarded_insertion_sort
            ins(i)
    return np.array(ix, np.int64)


def serialize(coord, grid_size=0.005, orders=ORDERS, tie="std"):
    """Point.serialization(order=orders, shuffle_orders=False) for one cloud -> dict(grid (N,3), code/order/inverse (len(orders), N))."""
    g = grid_coord(coord, grid_size)
    depth = int(g.max()).bit_length()
    assert depth * 3 <= 63 and depth <= 16
    code = np.stack([encode(g, depth, o) for o in orders])
    order = np.stack([std_argsort(c) for c in code]) if tie == "std" else np.argsort(code, axis=1, kind="stable")
    inverse = np.empty_like(order)
    np.put_along_axis(inverse, order, np.broadcast_to(np.arange(code.shape[1]), order.shape), 1)
    return {"grid": g, "code": code, "order": order, "inverse": inverse}
