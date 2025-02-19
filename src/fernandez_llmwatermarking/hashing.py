def get_seed_rng(
    start, 
    input_ids: list[int],
    salt = 35317
) -> int:
    """
    Seed RNG with hash of input_ids.
    Adapted from https://github.com/jwkirchenbauer/lm-watermarking
    """
    for ii in input_ids:
        start = (start * salt + ii) % (2 ** 64 - 1)
    return int(start)