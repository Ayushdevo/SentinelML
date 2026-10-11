"""Validate fitted model metadata before using it to select prediction inputs."""
from numbers import Real


def positive_class_index(model):
    try:
        classes = list(getattr(model, "classes_", (0, 1)))
    except TypeError as exc:
        raise ValueError("Model must expose binary churn classes 0 and 1") from exc
    if (
        len(classes) != 2
        or any(isinstance(label, bool) or not isinstance(label, Real)
               or label not in (0, 1) for label in classes)
        or set(classes) != {0, 1}
    ):
        raise ValueError("Model must expose binary churn classes 0 and 1")
    return classes.index(1)


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
