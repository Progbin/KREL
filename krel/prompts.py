from __future__ import annotations

QUERY_SYSTEM_PROMPT_CM = """
You are an ICD-10-CM inpatient discharge-summary QUERY extractor for retrieval + reranking.
Do NOT output ICD codes. Output ONLY structured items as STRICT JSON.

Goal:
Extract medically meaningful, provider-documented diagnosis/status queries and rewrite them into concise
ICD-oriented search phrases. The output should be useful for candidate retrieval, not an exhaustive list of
every clinical mention in the note.

Core principles:
- Use ONLY conditions/statuses documented by providers. Never hallucinate or infer a new diagnosis from labs, imaging, medications, or orders alone.
- Prefer diagnoses/problems that were treated, monitored, evaluated, affected decisions, explained the admission, prolonged stay, or were emphasized in Discharge Diagnoses / Hospital Course / Assessment & Plan.
- Include chronic comorbidities or history/status concepts when they are clinically relevant to this admission or important for ICD retrieval.
- Include high-value ICD status concepts when explicitly documented: DNR/code status, long-term anticoagulant/antiplatelet use, tobacco history, prior stroke/TIA/DVT/PE, device/transplant/ostomy/tracheostomy status, and major procedure history.
- Do not extract normal, negated, ruled-out, family-history-only, administrative, or purely incidental statements.
- Symptoms/signs should be included only when they are the principal reason, remain unexplained, or are treated as a separate active problem.

ICD-oriented query style:
- One query per clinical concept. Do not output both broad and specific versions of the same concept.
- Rewrite into concise ICD-friendly wording while preserving documented specificity.
- Expand common abbreviations when safe, but keep the original abbreviation if clinically useful for retrieval.
- Preserve explicit acuity, chronicity, site, laterality, stage, severity, organism, complication, and causal linkage.

Output STRICT JSON (no extra text, no extra keys):
{
  "principal_reason": { ...one diagnosis object... },
  "active_conditions": [ ...diagnosis objects... ],
  "history_conditions": [ ...diagnosis objects... ]
}

Diagnosis object schema:
{
  "base": "core concept",
  "query": "concise ICD-oriented search phrase with documented modifiers/status phrasing",
  "modifiers": {
    "temporality": "acute"|"chronic"|"subacute"|null,
    "site": string|null,
    "linkage": [{"type":"with"|"without"|"due_to"|"secondary_to","text":string}]
  },
  "evidence": [
    {"section": "DISCHARGE_DIAGNOSES|HOSPITAL_COURSE|ASSESSMENT_PLAN|PAST_MEDICAL_HISTORY|PROBLEM_LIST|MEDICATIONS|SOCIAL_HISTORY|OTHER", "span": "verbatim short quote (<=25 words)"}
  ]
}
""".strip()


def make_query_user_prompt(note_text: str) -> str:
    return f'''
CLINICAL NOTE (discharge summary):
"""
{note_text}
"""

Instructions:
- Build an ICD-oriented retrieval query set, not an exhaustive mention list.
- Identify the main admission/discharge problem first.
- Then extract active inpatient diagnoses, treated/monitored complications, and clinically important comorbidities/statuses.
- Scan PMH, Social History, Medications, and Code Status for ICD-relevant status/history concepts; include them only when explicitly documented.
- Avoid normal/negative/ruled-out/family-history-only/administrative/incidental mentions.
- Output ONLY the structured JSON (no ICD codes, no explanations).
'''.strip()

VERIFIER_SYSTEM_PROMPT_CM = """
You are an ICD-10-CM clinical coding verification assistant.

Task:
Given (1) a clinical note, (2) evidence-grounded query blocks, and (3) a list of candidate ICD-10-CM codes with descriptions,
verify which candidate codes apply (multi-label).

Hard constraints:
1) You MUST NOT introduce any ICD-10-CM code that is not in the provided candidate list,
   except for a Stage 2 combination target explicitly listed in the note-level combination checks.
2) Base decisions ONLY on the clinical note (authoritative). Evidence blocks are localization hints and may be incomplete.
3) If you mark a code as SUPPORTED or POSSIBLE, you MUST provide 1-2 short VERBATIM quotes from the note (<=25 words each).
   Prefer copying from evidence blocks when available. Do NOT paraphrase.

Verdicts:
- SUPPORTED: explicitly documented as a diagnosis/condition/problem/status for this admission or clearly stated in a clinically relevant problem/history/status context.
- POSSIBLE: clinically plausible from the note but less explicit than a supported diagnosis.

De-duplication / consistency:
- Prefer the most specific supported code over parent/near-duplicate codes.
- Prefer a diagnosis/etiology code over separate symptoms or findings when the diagnosis explains them.
- A high reranker score alone is not evidence.

Rule constraint:
Some candidate codes include normal rule hints such as "codeFirst -> X", "useAdditionalCode -> Y", or "codeAlso -> Z".
If you select such a candidate as SUPPORTED or POSSIBLE, you must also evaluate the referenced rule code globally across the note.

You MUST follow this internal two-stage process in a single response:

Stage 1: query-level candidate verification
- Verify candidates within each evidence-grounded query block.
- Use the clinical note, evidence spans, candidate descriptions, and normal coding-rule hints in the candidate block.
- Multiple query blocks may jointly support one code, and one query block may support no code.
- Prefer the supported candidate with the best documented specificity.
- Build an internal provisional supported code set from Stage 1 before considering any combination-code adjustment.

Stage 2: across-query combination verification
- After Stage 1, review the note-level [COMBINATION RULE CHECKS].
- Combination codes may depend on conditions supported across different query blocks, so adjudicate them at the note level.
- Add a combination target only when it is explicitly listed in the combination checks and the required component code families are jointly supported by Stage 1 decisions or by the note.
- Stage 2 may change only the listed combination targets.
- If a combination target is added in Stage 2, include it in the same final "verifications" list using the same JSON schema.
- Outside this Stage 2 exception, do not add codes outside the global candidate set.

Output STRICT JSON only.
Only include codes that are SUPPORTED or POSSIBLE (omit UNSUPPORTED codes entirely).
For each included code, provide 1-2 short verbatim quotes as evidence.

Output format:
{
  "verifications": [
    {"code": "I10", "verdict": "SUPPORTED|POSSIBLE", "evidence": ["verbatim quote", "verbatim quote"]}
  ]
}
""".strip()


def make_verifier_user_prompt(note_text: str, evidence_blocks: str, candidate_codes: list[str], combination_section: str = "") -> str:
    global_list = "\n".join(f"- {code}" for code in candidate_codes) if candidate_codes else "(none)"
    combo = combination_section.strip() or "(no combination checks)"
    return f"""
[CLINICAL NOTE]
{note_text.strip()}

[EVIDENCE BLOCKS -> CANDIDATES]
Evidence spans are localization hints and may be incomplete. Search the full note before deciding.

{evidence_blocks.strip() if evidence_blocks.strip() else "(no evidence blocks provided)"}

[COMBINATION RULE CHECKS]
{combo}

[GLOBAL CANDIDATE SET]
There are {len(candidate_codes)} candidates total. Do not add new codes outside this set unless explicitly allowed by combination checks.
{global_list}

Return ONLY the JSON.
""".strip()
