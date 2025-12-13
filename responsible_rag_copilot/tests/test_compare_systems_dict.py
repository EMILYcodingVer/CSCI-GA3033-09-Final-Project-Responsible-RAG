import compare_systems as cs


def test_compare_systems_dict_structure(monkeypatch):
    # Mock the expensive / external parts
    monkeypatch.setattr(cs, "answer_llm_only", lambda q: "BASELINE_ANSWER")
    monkeypatch.setattr(cs, "retrieve_docs", lambda q, k=3: [{"title": "Doc", "content": "Text"}])
    monkeypatch.setattr(cs, "format_docs", lambda docs: "FORMATTED")
    monkeypatch.setattr(cs, "call_llm", lambda prompt: "RAG_ANSWER")
    monkeypatch.setattr(cs, "run_pipeline", lambda q, k=3: {"final_answer": "RESP_ANSWER", "critic": "..."})

    out = cs.compare_systems_dict("Q", k=2)

    assert out["query"] == "Q"
    assert out["k"] == 2
    assert out["baseline"]["final_answer"] == "BASELINE_ANSWER"
    assert out["rag"]["final_answer"] == "RAG_ANSWER"
    assert isinstance(out["rag"]["sources"], list)
    assert out["responsible"]["final_answer"] == "RESP_ANSWER"
