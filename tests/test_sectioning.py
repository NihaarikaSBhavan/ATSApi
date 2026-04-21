from atsv4.sectioning import section_completeness_score, split_into_sections


def test_split_into_sections_matches_fuzzy_headers() -> None:
    text = """
    Profile
    Data scientist with NLP experience
    Work Experience
    Built ranking systems
    Tech Stack
    Python, PyTorch
    Education
    B.Tech Computer Science
    """
    sections = split_into_sections(text)

    assert "NLP experience" in sections["summary"]
    assert "ranking systems" in sections["experience"]
    assert "PyTorch" in sections["skills"]
    assert "B.Tech" in sections["education"]


def test_section_completeness_score_counts_core_sections() -> None:
    score = section_completeness_score(
        {
            "summary": "x",
            "experience": "y",
            "skills": "z",
            "education": "",
        }
    )
    assert score == 0.75
