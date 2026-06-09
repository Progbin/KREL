from __future__ import annotations

import re


SECTION_RULES: list[tuple[str, list[str]]] = [
    ("CHIEF_COMPLAINT", ["Chief Complaint", "CC"]),
    ("HPI", ["History of Present Illness", "HPI"]),
    ("PMH", ["Past Medical History", "PMH", "Medical History"]),
    ("PSH", ["Past Surgical History", "PSH"]),
    ("PROBLEM_LIST", ["Problem List", "Problems"]),
    ("MEDICATIONS", ["Medications", "Home Medications", "Discharge Medications"]),
    ("SOCIAL_HISTORY", ["Social History"]),
    ("FAMILY_HISTORY", ["Family History"]),
    ("PHYSICAL_EXAM", ["Physical Exam", "Physical Examination"]),
    ("LABS", ["Labs", "Laboratory Data", "Laboratory"]),
    ("IMAGING", ["Imaging", "Radiology", "Studies"]),
    ("HOSPITAL_COURSE", ["Hospital Course"]),
    ("ASSESSMENT_PLAN", ["Assessment and Plan", "Assessment/Plan", "A/P"]),
    ("DISCHARGE_DIAGNOSES", ["Discharge Diagnosis", "Discharge Diagnoses"]),
]


def _build_patterns() -> list[tuple[str, re.Pattern[str]]]:
    patterns: list[tuple[str, re.Pattern[str]]] = []
    for section, headers in SECTION_RULES:
        escaped = []
        for header in headers:
            parts = [re.escape(part) for part in header.split()]
            escaped.append(r"\s+".join(parts))
        pattern = re.compile(rf"^\s*(?:{'|'.join(escaped)})\s*:?\s*$", flags=re.IGNORECASE)
        patterns.append((section, pattern))
    return patterns


HEADER_PATTERNS = _build_patterns()


def sectionize_text(text: str) -> str:
    """Insert explicit section tags before common MIMIC-style note headers."""
    if not text:
        return ""
    out_lines: list[str] = []
    for line in str(text).splitlines():
        for section, pattern in HEADER_PATTERNS:
            if pattern.match(line):
                out_lines.append(f"[SECTION: {section}]")
                break
        out_lines.append(line)
    return "\n".join(out_lines)
