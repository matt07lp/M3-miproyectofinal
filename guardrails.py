# ============================================================
# Cell 8 - Input & Output Guardrails (Improved)
# ============================================================
import re

# ── Restricted patterns (prompt injection) ──
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"forget\s+(your\s+)?(system\s+)?prompt",
    r"you\s+are\s+now",
    r"^SYSTEM\s*:",
    r"override",
    r"return\s+(all\s+)?(internal|system)\s+prompts?",
    r"disregard",
    r"pretend\s+you",
    r"act\s+as\s+(?!a\s+financial)",   # "act as X" unless "act as a financial..."
]

# ── Off-topic patterns ──
OFFTOPIC_PATTERNS = [
    r"\b(recipe|cook|bake|ingredient)\b",
    r"\b(poem|poetry|song|story|write\s+me)\b",
    r"\b(world\s+cup|football|soccer|basketball|nba|nfl)\b",
    r"\b(movie|film|actor|actress|celebrity)\b",
    r"\b(weather|forecast)\b",
    r"\bwho\s+won\b(?!.*\b(contract|bid|award)\b)",  # "who won" unless business context
]

# ── SQL injection patterns ──
SQL_PATTERNS = [
    r"\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC)\b.*\b(FROM|INTO|TABLE|DATABASE)\b",
    r";\s*(DROP|DELETE|ALTER|EXEC)\b",
    r"--\s*$",
    r"\bUNION\s+SELECT\b",
]

# ── Financial relevance keywords ──
FINANCIAL_KEYWORDS = [
    r"\b(revenue|income|profit|loss|earnings|eps)\b",
    r"\b(debt|liability|liabilities|asset|assets|equity)\b",
    r"\b(expense|cost|margin|cash\s+flow|capex)\b",
    r"\b(10-[KQ]|10K|10Q|SEC|filing|annual\s+report)\b",
    r"\b(balance\s+sheet|income\s+statement|financial)\b",
    r"\b(risk\s+factor|dividend|share|stock|fiscal)\b",
    r"\b(consolidat|operat|quarter|segment)\b",
]


def input_guardrail(query: str):
    """
    Returns (is_blocked: bool, reason: str).
    True = blocked, False = allowed.
    """
    # 1. Empty check
    if not query or not query.strip():
        return (True, "Empty query.")

    q_lower = query.strip().lower()

    # 2. SQL injection check
    for pattern in SQL_PATTERNS:
        if re.search(pattern, query, re.IGNORECASE):
            return (True, "Blocked: SQL injection attempt detected.")

    # 3. Prompt injection check
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, q_lower):
            return (True, "Blocked: matches restricted pattern.")

    # 4. Off-topic check
    for pattern in OFFTOPIC_PATTERNS:
        if re.search(pattern, q_lower):
            # Double-check: is it also financial?
            is_financial = any(re.search(fp, q_lower) for fp in FINANCIAL_KEYWORDS)
            if not is_financial:
                return (True, "Blocked: off-topic query (not related to financial filings).")

    # 5. Passed all checks
    return (False, "")


def output_guardrail(response):
    """
    Validates the generated response.
    Returns (is_blocked: bool, reason: str).
    True = blocked, False = allowed.
    """
    warnings = []

    # Handle string responses (model refused or raw text)
    if isinstance(response, str):
        response_lower = response.lower()
        # Check if it's a refusal — that's acceptable
        if "do not have sufficient information" in response_lower:
            return (False, "")
        # Otherwise, non-JSON output is suspicious
        warnings.append("Response is not structured JSON.")
        return (True, "; ".join(warnings))

    # Handle dict responses
    if isinstance(response, dict):
        answer = response.get("answer", "")
        confidence = response.get("confidence", "")
        sources = response.get("sources", [])

        # Validate confidence
        valid_confidences = {"high", "medium", "low"}
        if str(confidence).lower() not in valid_confidences:
            warnings.append(f"Unexpected confidence value: '{confidence}'.")

        # Check for empty answer
        if not answer or (isinstance(answer, str) and not answer.strip()):
            warnings.append("Empty answer.")

        # Check for hallucination red flags
        hallucination_flags = [
            "for demonstration purposes",
            "let's assume",
            "placeholder",
            "example",
        ]
        answer_lower = str(answer).lower()
        for flag in hallucination_flags:
            if flag in answer_lower:
                warnings.append(f"Possible hallucination detected: '{flag}'.")
                break

        if warnings:
            return (True, "; ".join(warnings))

        return (False, "")

    # Unknown type
    return (True, f"Unexpected response type: {type(response)}")