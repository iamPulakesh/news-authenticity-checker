from app.models.verdict import FactCheckVerdict, VerdictLabel, ClaimAnalysis


def test_verdict_label_values():
    assert VerdictLabel.REAL.value == "Real"
    assert VerdictLabel.FAKE.value == "Fake"
    assert VerdictLabel.MISLEADING.value == "Misleading"
    assert VerdictLabel.UNVERIFIED.value == "Unverified"


def test_default_verdict():
    v = FactCheckVerdict()
    assert v.verdict == VerdictLabel.UNVERIFIED
    assert v.confidence_score == 0.0
    assert v.claims_analyzed == []
    assert v.sources_consulted == []


def test_verdict_emoji():
    assert FactCheckVerdict(verdict=VerdictLabel.REAL).verdict_emoji() == "REAL"
    assert FactCheckVerdict(verdict=VerdictLabel.FAKE).verdict_emoji() == "FAKE"
    assert FactCheckVerdict(verdict=VerdictLabel.MISLEADING).verdict_emoji() == "MISLEADING"
    assert FactCheckVerdict(verdict=VerdictLabel.UNVERIFIED).verdict_emoji() == "UNVERIFIED"


def test_confidence_bar():
    assert FactCheckVerdict(confidence_score=0.85).confidence_bar() == "85%"
    assert FactCheckVerdict(confidence_score=0.0).confidence_bar() == "0%"
    assert FactCheckVerdict(confidence_score=1.0).confidence_bar() == "100%"


def test_to_dict():
    claim = ClaimAnalysis(claim="Test claim", status="Supported", confidence="High", evidence="Some evidence")
    v = FactCheckVerdict(
        verdict=VerdictLabel.FAKE,
        confidence_score=0.9,
        claims_analyzed=[claim],
        reasoning_summary="Test summary",
        sources_consulted=["https://example.com"],
        article_title="Test Article",
    )
    d = v.to_dict()

    assert d["verdict"] == "Fake"
    assert d["confidence_score"] == 0.9
    assert len(d["claims_analyzed"]) == 1
    assert d["claims_analyzed"][0]["claim"] == "Test claim"
    assert d["reasoning_summary"] == "Test summary"
    assert d["sources_consulted"] == ["https://example.com"]
    assert d["article_title"] == "Test Article"


def test_claim_analysis_defaults():
    c = ClaimAnalysis()
    assert c.claim == "Unknown claim"
    assert c.status == "Unverifiable"
    assert c.confidence == "Low"
    assert c.evidence == "No evidence provided"
