"""Validate fitted model metadata before using it to select prediction inputs."""


def model_feature_order(model, expected):
    expected = list(expected)
    if not hasattr(model, "feature_names_in_"):
        return expected
    try:
        names = list(model.feature_names_in_)
    except TypeError as exc:
        raise ValueError("Model feature names do not match prediction schema") from exc
    if (
        len(names) != len(expected)
        or not all(isinstance(name, str) for name in names)
        or set(names) != set(expected)
    ):
        raise ValueError("Model feature names do not match prediction schema")
    return names
