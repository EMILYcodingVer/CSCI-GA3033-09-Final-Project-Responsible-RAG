import types

import answer_pipeline


def test_pipeline_runs_with_mocked_llm(monkeypatch):
    # Mock the LLM call to avoid network and make outputs deterministic.
    def fake_chat(*args, **kwargs):
        return "FAKE_RESPONSE"

    monkeypatch.setattr(answer_pipeline, "chat", fake_chat, raising=False)

    # Minimal call: verify the pipeline returns expected keys.
    out = answer_pipeline.run_responsible_pipeline(
        user_query="test query",
        mode="RESPONSIBLE",
        retrieved_docs=[{"id": "0", "text": "doc"}],
    )

    assert isinstance(out, dict)
    assert "final_answer" in out
    assert "critic" in out
