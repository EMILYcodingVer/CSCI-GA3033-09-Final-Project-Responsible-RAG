# filters.py
# Simple placeholder filter functions

def hallucination_check(query, docs, draft):
    """
    Dummy hallucination check function.
    Always returns a neutral result.
    """
    return "No hallucination detected (placeholder)."


def safety_filter(answer):
    """
    Dummy safety filter function.
    Always returns the answer directly.
    """
    return answer