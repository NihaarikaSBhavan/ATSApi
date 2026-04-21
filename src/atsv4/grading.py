from __future__ import annotations


def to_percentage(score_0_to_1: float) -> float:
    return round(max(0.0, min(score_0_to_1, 1.0)) * 100, 2)


def to_grade(score_percent: float) -> str:
    if score_percent >= 85:
        return "A"
    if score_percent >= 70:
        return "B"
    if score_percent >= 55:
        return "C"
    if score_percent >= 40:
        return "D"
    return "E"


def grade_to_score(grade: str) -> float:
    normalized_grade = grade.strip().upper()
    return {
        "A": 92.0,
        "B": 78.0,
        "C": 62.0,
        "D": 45.0,
        "E": 25.0,
    }.get(normalized_grade, 25.0)
