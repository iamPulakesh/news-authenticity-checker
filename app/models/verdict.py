from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class VerdictLabel(Enum):
    """Possible fact-check outcomes."""
    REAL        = "Real"
    FAKE        = "Fake"
    MISLEADING  = "Misleading"
    UNVERIFIED  = "Unverified"


@dataclass
class ClaimAnalysis:

    claim:      str = "Unknown claim"
    status:     str = "Unverifiable"
    confidence: str = "Low"
    evidence:   str = "No evidence provided"


@dataclass
class FactCheckVerdict:
    """
    The final structured output of the fact-checking agent.
    Produced by verdict_generator_node in graph.py and returned
    to the frontend.
    """
    verdict:            VerdictLabel          = VerdictLabel.UNVERIFIED
    confidence_score:   float                 = 0.0
    claims_analyzed:    List[ClaimAnalysis]    = field(default_factory=list)
    reasoning_summary:  str                   = ""
    sources_consulted:  List[str]             = field(default_factory=list)
    cot_steps:          str                   = ""
    input_type:         str                   = "unknown"
    article_title:      str                   = ""

    def verdict_emoji(self) -> str:

        label_map = {
            VerdictLabel.REAL:       "REAL",
            VerdictLabel.FAKE:       "FAKE",
            VerdictLabel.MISLEADING: "MISLEADING",
            VerdictLabel.UNVERIFIED: "UNVERIFIED",
        }
        return label_map.get(self.verdict, "UNKNOWN")

    def confidence_bar(self) -> str:
        """Return the confidence as a percentage string e.g. '80%'."""
        return f"{self.confidence_score:.0%}"

    def to_dict(self) -> dict:

        return {
            "verdict":           self.verdict.value,
            "confidence_score":  self.confidence_score,
            "claims_analyzed": [
                {
                    "claim":      c.claim,
                    "status":     c.status,
                    "confidence": c.confidence,
                    "evidence":   c.evidence,
                }
                for c in self.claims_analyzed
            ],
            "reasoning_summary": self.reasoning_summary,
            "sources_consulted": self.sources_consulted,
            "cot_steps":         self.cot_steps,
            "input_type":        self.input_type,
            "article_title":     self.article_title,
        }
