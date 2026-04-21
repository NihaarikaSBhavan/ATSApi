from __future__ import annotations

from rapidfuzz import fuzz

CANONICAL_SECTION_ALIASES: dict[str, tuple[str, ...]] = {
    "summary": ("summary", "profile", "professional summary", "objective", "about"),
    "experience": (
        "experience",
        "work experience",
        "employment",
        "professional experience",
        "career history",
    ),
    "skills": ("skills", "technical skills", "tech stack", "competencies", "expertise"),
    "education": ("education", "academic background", "qualifications"),
}


def split_into_sections(text: str, min_header_score: float = 82.0) -> dict[str, str]:
    sections: dict[str, list[str]] = {name: [] for name in CANONICAL_SECTION_ALIASES}
    sections["other"] = []
    current_section = "other"

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        detected = _match_section_header(line, min_header_score)
        if detected is not None:
            current_section = detected
            continue

        sections[current_section].append(line)

    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def section_completeness_score(sections: dict[str, str]) -> float:
    required = ("summary", "experience", "skills", "education")
    present = sum(1 for name in required if sections.get(name, "").strip())
    return present / len(required)


def _match_section_header(line: str, min_header_score: float) -> str | None:
    normalized = line.lower().strip(":").strip()
    if len(normalized) > 40:
        return None

    best_section: str | None = None
    best_score = 0.0
    for section_name, aliases in CANONICAL_SECTION_ALIASES.items():
        for alias in aliases:
            score = fuzz.ratio(normalized, alias)
            if score > best_score:
                best_score = score
                best_section = section_name

    return best_section if best_score >= min_header_score else None
