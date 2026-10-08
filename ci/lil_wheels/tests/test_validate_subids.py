import pytest

from ci.lil_wheels.validate_subids import validate_ranges


@pytest.mark.parametrize(
    "text",
    [
        "runner:200000:65536",
        "runner:200000:32768\nrunner:300000:32768",
        "other:100000:65536\nrunner:200000:65536",
    ],
)
def test_valid_mapping(text):
    validate_ranges(text, "runner")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "runner:200000:65535",
        "runner:200000:65536\nother:200010:2",
        "runner:200000:65536\nrunner:200010:2",
        "runner:200000:0",
        "runner:4294967290:65536",
        "\n".join(f"runner:{200000 + i * 20000}:12000" for i in range(6)),
    ],
)
def test_invalid_mapping(text):
    with pytest.raises(ValueError):
        validate_ranges(text, "runner")


def test_candidate_range_must_be_unused():
    validate_ranges("other:100000:65536", "runner", candidate=True)
    with pytest.raises(ValueError, match="overlap"):
        validate_ranges("other:200000:65536", "runner", candidate=True)
