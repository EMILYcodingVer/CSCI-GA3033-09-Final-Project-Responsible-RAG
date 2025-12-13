import types

from scripts.experiment_loan_approval import simple_trust_metrics


def test_simple_trust_metrics_baseline():
    m = simple_trust_metrics("baseline", {"retrieved_docs": [], "critic": None})
    assert m.source_verifiability == "X"
    assert m.risk_visibility == "X"


def test_simple_trust_metrics_rag():
    m = simple_trust_metrics("rag", {"retrieved_docs": ["doc"], "critic": None})
    assert m.source_verifiability.startswith("✓")


def test_simple_trust_metrics_responsible():
    m = simple_trust_metrics("responsible", {"retrieved_docs": ["doc"], "critic": {"suggestions": ["x"]}})
    assert m.risk_visibility.startswith("✓")
    assert m.actionability.startswith("✓")
