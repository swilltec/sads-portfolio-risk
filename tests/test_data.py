from pathlib import Path

import numpy as np
import pytest
from sads.data import load_returns


def test_french_parser_uses_value_weighted_block_and_decimal_units(
    tmp_path: Path,
) -> None:
    path = tmp_path / "french.csv"
    names = ",".join(f"industry{i}" for i in range(49))
    values = ",".join(["1.25"] * 49)
    path.write_text(
        f"Preamble\n  Average Value Weighted Returns -- Daily\n,{names}\n"
        f"20000103,{values}\n\n  Average Equal Weighted Returns -- Daily\n"
        f",{names}\n20000103,{values}\n"
    )
    frame = load_returns(path)
    assert frame.shape == (1, 49)
    np.testing.assert_allclose(frame.to_numpy(), 0.0125)
    path.write_text(path.read_text().replace("1.25", "-99.99", 1))
    with pytest.raises(ValueError, match="incomplete"):
        load_returns(path)
