import metrics


def test_profit() -> None:
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    assert metrics.profit([1, 2, 3], [1, 1, 1]) == 3


def test_margin() -> None:
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    assert metrics.margin([1, 2, 3], [1, 1, 1]) == 0.5


def test_markup() -> None:
    """Form a complex number.

    Keyword arguments:
    real -- the real part (default 0.0)
    imag -- the imaginary part (default 0.0)
    """
    assert metrics.markup([1, 2, 3], [1, 1, 1]) == 1
