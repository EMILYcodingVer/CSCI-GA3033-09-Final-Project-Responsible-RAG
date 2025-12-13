import compare_systems as cs


def test_compute_trust_metrics_monotonicity():
    baseline = cs.compute_trust_metrics(
        mode="baseline",
        has_sources=False,
        has_snapshot=False,
        has_critic=False,
        has_suggestions=False,
        structured_answer=False,
    )
    rag = cs.compute_trust_metrics(
        mode="rag",
        has_sources=True,
        has_snapshot=False,
        has_critic=False,
        has_suggestions=False,
        structured_answer=True,
    )
    responsible = cs.compute_trust_metrics(
        mode="responsible",
        has_sources=True,
        has_snapshot=True,
        has_critic=True,
        has_suggestions=True,
        structured_answer=True,
    )

    assert baseline["evidence_transparency"] <= rag["evidence_transparency"] <= responsible["evidence_transparency"]
    assert baseline["risk_clarity"] <= rag["risk_clarity"] <= responsible["risk_clarity"]
    assert baseline["action_value"] <= rag["action_value"] <= responsible["action_value"]
    assert baseline["trust_index"] < responsible["trust_index"]
