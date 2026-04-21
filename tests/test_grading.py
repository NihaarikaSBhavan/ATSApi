from atsv4.grading import to_grade, to_percentage


def test_to_percentage_clamps_values() -> None:
    assert to_percentage(1.2) == 100.0
    assert to_percentage(-0.1) == 0.0


def test_to_grade_bands() -> None:
    assert to_grade(87) == "A"
    assert to_grade(72) == "B"
    assert to_grade(58) == "C"
    assert to_grade(44) == "D"
    assert to_grade(20) == "E"
