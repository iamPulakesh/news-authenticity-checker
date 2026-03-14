import os
import pytest
from app.agent.graph import build_agent_graph
from app.agent.tools import source_checker_tool, rag_search_tool
from app.config import settings


class TestAgentGraph:

    def test_graph_compiles(self):
        agent = build_agent_graph()
        assert agent is not None

    def test_graph_has_nodes(self):
        agent = build_agent_graph()
        node_names = list(agent.get_graph().nodes.keys())
        assert "input_router" in node_names
        assert "claim_extractor" in node_names
        assert "evidence_retriever" in node_names
        assert "verdict_generator" in node_names


class TestSourceChecker:

    def test_known_credible_source(self):
        result = source_checker_tool.invoke({"domain": "bbc.com"})
        assert "SOURCE_CREDIBILITY" in result
        assert "0.95" in result

    def test_known_unreliable_source(self):
        result = source_checker_tool.invoke({"domain": "infowars.com"})
        assert "SOURCE_CREDIBILITY" in result
        assert "0.05" in result

    def test_www_prefix_stripped(self):
        result = source_checker_tool.invoke({"domain": "www.bbc.com"})
        assert "0.95" in result


class TestRAGTool:

    @pytest.mark.skipif(
        not os.path.exists("./data/vectorstore"),
        reason="Vectorstore not available"
    )
    def test_rag_search_returns_results(self):
        result = rag_search_tool.invoke({"query": "vaccines cause autism"})
        assert "RAG_RESULTS" in result or "RAG_EMPTY" in result


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set"
)
class TestEndToEnd:

    def test_full_factcheck_url(self):
        from app.agent.runner import run_fact_check
        verdict = run_fact_check("https://apnews.com/", verbose=False)
        assert verdict is not None
        assert verdict.verdict is not None
        assert 0.0 <= verdict.confidence_score <= 1.0
