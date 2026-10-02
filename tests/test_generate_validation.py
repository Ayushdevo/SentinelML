"""Python bools are integers, but not valid dataset sizes."""
import pytest

from src.generate_data import build_dataset


@pytest.mark.parametrize("bad_size", [True, False, 0, 9, 1.5, "100"])
def test_invalid_sample_sizes_fail_explicitly(bad_size):
    with pytest.raises(ValueError, match="n_samples"):
        build_dataset(bad_size)
