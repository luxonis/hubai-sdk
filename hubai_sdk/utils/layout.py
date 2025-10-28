def make_default_layout(shape: list[int]) -> str:
    """Creates a default layout for the given shape.

    Tries to guess most common layouts for the given shape pattern.
    Otherwise, uses the first free letter of the alphabet for each dimension.

    Example:
        >>> make_default_layout([1, 3, 256, 256])
        >>> "NCHW"
        >>> make_default_layout([1, 19, 7, 8])
        >>> "NABC"
    """
    layout = []
    i = 0
    if shape[0] == 1:
        layout.append("N")
        i += 1
    if len(shape) - i == 3:
        if shape[i] < shape[i + 1] and shape[i] < shape[i + 2]:
            return "".join([*layout, "C", "H", "W"])
        if shape[-1] < shape[-2] and shape[-1] < shape[-3]:
            return "".join([*layout, "H", "W", "C"])
    i = 0
    while len(layout) < len(shape):
        # Starting with "C" for more sensible defaults
        letter = chr(ord("A") + (i + 2) % 26)
        if letter not in layout:
            layout.append(letter)
        i += 1
    return "".join(layout)
