"""Analysis and metric calculation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricResult:
    """Standardized result container for all metrics."""

    name: str
    raw_value: float
    normalized_score: float
    human_standard: float
    ai_standard: float
    verdict: str
    details: Dict
    warning: Optional[str] = None


@dataclass
class TextStats:
    """Pre-computed text statistics used across metrics."""

    text: str
    words: List[str]
    sentences: List[str]
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    pos_tags: List[Tuple[str, str]]
    is_short_text: bool


@dataclass
class AnalysisResult:
    score: float
    classification: str
    metrics_edited: Dict[str, float]
    metrics_original: Dict[str, float]
    components: Dict[str, float]
    metric_standards: Dict[str, Dict[str, float]]
    metric_results: Dict[str, MetricResult]
    consistency_score: float
    word_delta: int
    sentence_delta: int
    ai_ism_total: int
    ai_ism_categories: Dict[str, int]
    ai_ism_categories_original: Dict[str, int]
    ai_ism_phrases: List[Dict]
    sentence_lengths: Dict[str, List[int]]
    original_word_count: int
    edited_word_count: int
    original_sentence_count: int
    edited_sentence_count: int


class CalibrationStandards:
    """Default and adjustable standards for human vs AI writing."""

    DEFAULTS = {
        "burstiness": {"human": 1.23, "ai": 0.78},
        "lexical_diversity": {"human": 0.55, "ai": 0.42},
        "syntactic_complexity": {"human": 0.54, "ai": 0.64},
        "ai_ism_likelihood": {"human": 3.1, "ai": 78.5},
        "function_word_ratio": {"human": 0.50, "ai": 0.60},
        "discourse_marker_density": {"human": 8.0, "ai": 18.0},
        "information_density": {"human": 0.58, "ai": 0.42},
        "epistemic_hedging": {"human": 0.09, "ai": 0.04},
    }

    def __init__(self, custom_standards: Optional[Dict] = None):
        self.standards = custom_standards or self.DEFAULTS.copy()

    def get(self, metric: str) -> Dict[str, float]:
        return self.standards.get(metric, self.DEFAULTS[metric])


class TextPreprocessor:
    """Handles all text preprocessing and statistical computation."""

    SHORT_TEXT_THRESHOLD = 150

    FUNCTION_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "as",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "i",
        "you",
        "he",
        "she",
        "it",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "mine",
        "yours",
        "hers",
        "ours",
        "theirs",
        "what",
        "which",
        "who",
        "when",
        "where",
        "why",
        "how",
        "all",
        "each",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "nor",
        "not",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "now",
        "then",
        "here",
        "there",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "once",
        "during",
        "before",
        "after",
        "above",
        "below",
        "between",
        "through",
        "into",
        "within",
        "without",
        "against",
        "about",
    }

    CONTENT_POS = {
        "NN",
        "NNS",
        "NNP",
        "NNPS",
        "VB",
        "VBD",
        "VBG",
        "VBN",
        "VBP",
        "VBZ",
        "JJ",
        "JJR",
        "JJS",
        "RB",
        "RBR",
        "RBS",
    }

    def preprocess(self, text: str) -> TextStats:
        text = text.strip()
        if not text:
            raise ValueError("Empty text provided")

        sentences = self._sentence_tokenize(text)
        words = self._word_tokenize(text.lower())
        alpha_words = [w for w in words if any(c.isalpha() for c in w)]

        word_count = len(alpha_words)
        sentence_count = len(sentences)

        if sentence_count == 0:
            raise ValueError("No sentences detected in text")

        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        pos_tags = self._simple_pos_tag(alpha_words)
        is_short_text = word_count < self.SHORT_TEXT_THRESHOLD

        return TextStats(
            text=text,
            words=alpha_words,
            sentences=sentences,
            word_count=word_count,
            sentence_count=sentence_count,
            avg_sentence_length=avg_sentence_length,
            pos_tags=pos_tags,
            is_short_text=is_short_text,
        )

    def _sentence_tokenize(self, text: str) -> List[str]:
        pattern = r"(?<=[.!?])\s+(?=[A-Z])"
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _word_tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text)

    def _simple_pos_tag(self, words: List[str]) -> List[Tuple[str, str]]:
        tags = []
        for word in words:
            w = word.lower()
            if w in {"the", "a", "an"}:
                tag = "DT"
            elif w in {"is", "was", "are", "were", "be", "been", "being"}:
                tag = "VB"
            elif w in {"i", "you", "he", "she", "it", "we", "they"}:
                tag = "PRP"
            elif w in {"in", "on", "at", "to", "for", "of", "with"}:
                tag = "IN"
            elif w.endswith("ing"):
                tag = "VBG"
            elif w.endswith("ed"):
                tag = "VBD"
            elif w.endswith("ly"):
                tag = "RB"
            elif w[0].isupper() and len(w) > 1:
                tag = "NNP"
            elif w in self.FUNCTION_WORDS:
                tag = "XX"
            else:
                tag = "NN"
            tags.append((word, tag))
        return tags


class BurstinessMetric:
    name = "Burstiness"
    MIN_SENTENCES = 3

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None
        sentence_lengths = []
        for sent in stats.sentences:
            length = len(re.findall(r"\b\w+\b", sent))
            if length > 0:
                sentence_lengths.append(length)

        if len(sentence_lengths) < self.MIN_SENTENCES:
            warning = f"Insufficient sentences ({len(sentence_lengths)}) for reliable burstiness"
            raw_value = 0.0
            normalized = 0.5
        else:
            mean_len = sum(sentence_lengths) / len(sentence_lengths)
            variance = sum((x - mean_len) ** 2 for x in sentence_lengths) / len(sentence_lengths)
            std_dev = math.sqrt(variance)
            raw_value = std_dev / mean_len if mean_len > 0 else 0.0
            std = standards.get("burstiness")
            normalized = self._normalize(raw_value, std["human"], std["ai"])

        verdict = self._get_verdict(normalized)

        if stats.is_short_text:
            warning = warning or "Short text: burstiness may be unreliable"

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=standards.get("burstiness")["human"],
            ai_standard=standards.get("burstiness")["ai"],
            verdict=verdict,
            details={"sentence_lengths": sentence_lengths},
            warning=warning,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if human_std == ai_std:
            return 0.5
        if value >= human_std:
            return min(1.0, 0.5 + 0.5 * (value - human_std) / human_std)
        if value <= ai_std:
            return max(0.0, 0.5 * (value / ai_std))
        return 0.5 + 0.5 * (value - ai_std) / (human_std - ai_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.7:
            return "Preserved"
        if normalized >= 0.4:
            return "Moderate"
        return "Compromised"


class LexicalDiversityMetric:
    name = "Lexical Diversity"
    TTR_THRESHOLD = 0.72
    MTLD_MIN_WORDS = 100

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None

        if stats.word_count < 50:
            warning = "Very short text: lexical diversity unreliable"

        if stats.word_count < self.MTLD_MIN_WORDS:
            raw_value = len(set(stats.words)) / max(stats.word_count, 1)
            method = "TTR"
        else:
            raw_value = self._calculate_mtld(stats.words)
            method = "MTLD"

        std = standards.get("lexical_diversity")
        normalized = self._normalize(raw_value, std["human"], std["ai"], method)
        verdict = self._get_verdict(normalized)

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={
                "method": method,
                "unique_words": len(set(stats.words)),
                "total_words": stats.word_count,
            },
            warning=warning,
        )

    def _calculate_mtld(self, words: List[str]) -> float:
        if len(words) < 10:
            return 0.0

        words = [w.lower() for w in words if w.isalpha()]

        def mtld_pass(word_list: List[str]) -> float:
            factors = 0
            types = set()
            token_count = 0

            for word in word_list:
                types.add(word)
                token_count += 1
                ttr = len(types) / token_count

                if ttr <= self.TTR_THRESHOLD:
                    factors += 1
                    types = set()
                    token_count = 0

            if token_count > 0 and len(types) / token_count < 1:
                factors += (1 - self.TTR_THRESHOLD) / (1 - len(types) / token_count)

            return len(word_list) / factors if factors > 0 else len(word_list)

        forward = mtld_pass(words)
        backward = mtld_pass(words[::-1])
        return (forward + backward) / 2

    def _normalize(self, value: float, human_std: float, ai_std: float, method: str) -> float:
        if method == "TTR":
            if value >= human_std:
                return min(1.0, value)
            return max(0.0, value / human_std * 0.5)

        if human_std <= ai_std:
            return 0.5
        if value >= human_std:
            return min(1.0, 0.5 + 0.5 * (value - human_std) / human_std)
        if value <= ai_std:
            return max(0.0, 0.5 * (value / ai_std))
        return (value - ai_std) / (human_std - ai_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.7:
            return "Preserved"
        if normalized >= 0.4:
            return "Moderate"
        return "Compromised"


class SyntacticComplexityMetric:
    name = "Syntactic Complexity"

    SUBORDINATORS = {
        "because",
        "since",
        "as",
        "although",
        "though",
        "while",
        "whereas",
        "if",
        "unless",
        "whether",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "after",
        "before",
        "until",
        "till",
        "once",
        "even",
        "provided",
        "assuming",
        "given",
    }

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None

        subord_count = 0
        for sent in stats.sentences:
            words = sent.lower().split()
            if any(w in self.SUBORDINATORS for w in words):
                subord_count += 1

        subord_ratio = subord_count / len(stats.sentences) if stats.sentences else 0
        length_factor = min(stats.avg_sentence_length / 30, 1.0)
        raw_value = (subord_ratio * 0.6) + (length_factor * 0.4)

        std = standards.get("syntactic_complexity")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        if stats.is_short_text:
            warning = "Short text: complexity patterns may not be representative"

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={"subordination_ratio": round(subord_ratio, 4)},
            warning=warning,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if human_std == ai_std:
            return 0.5
        if value <= human_std:
            return min(1.0, 0.5 + 0.5 * (human_std - value) / human_std)
        if value >= ai_std:
            return max(0.0, 0.5 * (ai_std / value) if value > 0 else 0)
        return 0.5 + 0.5 * (ai_std - value) / (ai_std - human_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.6:
            return "Preserved"
        if normalized >= 0.3:
            return "Moderate"
        return "Compromised"


class AIIsmMetric:
    name = "AI-ism Likelihood"

    AI_ISMS = {
        "connectors": {
            "moreover": 2,
            "furthermore": 2,
            "additionally": 2,
            "consequently": 2,
            "therefore": 2,
            "thus": 2,
            "hence": 2,
            "accordingly": 2,
            "in order to": 2,
            "as a result": 2,
            "in addition": 2,
            "on the other hand": 2,
            "in contrast": 2,
            "by contrast": 2,
            "in comparison": 2,
            "similarly": 2,
            "likewise": 2,
            "conversely": 2,
            "alternatively": 2,
            "subsequently": 2,
            "nevertheless": 2,
            "nonetheless": 2,
            "notwithstanding": 2,
            "regardless": 2,
        },
        "cliches": {
            "delve into": 3,
            "it is important to note": 3,
            "it is worth noting": 3,
            "it should be noted": 3,
            "plays a crucial role": 3,
            "plays an important role": 3,
            "sheds light on": 3,
            "offers insights into": 3,
            "provides insights into": 3,
            "a wide range of": 3,
            "a variety of": 3,
            "a number of": 3,
            "in the realm of": 3,
            "in the context of": 3,
            "in terms of": 3,
            "it is evident that": 3,
            "it is clear that": 3,
            "underscores the importance": 3,
            "highlights the significance": 3,
            "raises important questions": 3,
        },
        "generic_openers": {
            "in conclusion": 2,
            "to conclude": 2,
            "in summary": 2,
            "to summarize": 2,
            "in this essay": 2,
            "this essay will": 2,
            "this paper examines": 2,
            "the purpose of this": 2,
            "this study aims to": 2,
            "the present study": 2,
        }
    }

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None
        text_lower = stats.text.lower()
        total_weight = 0
        detected = []
        category_counts = {"connectors": 0, "cliches": 0, "generic_openers": 0}

        for category, phrases in self.AI_ISMS.items():
            for phrase, weight in phrases.items():
                count = text_lower.count(phrase)
                if count > 0:
                    category_counts[category] += count
                    total_weight += count * weight
                    detected.append(
                        {
                            "phrase": phrase,
                            "count": count,
                            "weight": weight,
                            "category": category,
                        }
                    )

        raw_value = (total_weight / max(stats.word_count, 1)) * 100

        std = standards.get("ai_ism_likelihood")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        if raw_value > 15:
            warning = "Very high AI-ism density suggests heavy AI editing"

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={
                "total_weighted_count": total_weight,
                "category_breakdown": category_counts,
                "detected_phrases": sorted(
                    detected,
                    key=lambda x: x["count"] * x["weight"],
                    reverse=True,
                )[:10],
            },
            warning=warning,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if ai_std <= human_std:
            return 0.5
        if value <= human_std:
            return 1.0
        if value >= ai_std:
            return 0.0
        return 1.0 - (value - human_std) / (ai_std - human_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.8:
            return "Preserved"
        if normalized >= 0.5:
            return "Moderate"
        return "Compromised"


class FunctionWordRatioMetric:
    name = "Function Word Ratio"

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        function_count = sum(1 for w in stats.words if w in TextPreprocessor.FUNCTION_WORDS)
        raw_value = function_count / max(stats.word_count, 1)

        std = standards.get("function_word_ratio")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={"function_words": function_count, "total_words": stats.word_count},
            warning=None,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if ai_std <= human_std:
            return 0.5
        if value <= human_std:
            return 1.0
        if value >= ai_std:
            return 0.0
        return 1.0 - (value - human_std) / (ai_std - human_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.7:
            return "Preserved"
        if normalized >= 0.4:
            return "Moderate"
        return "Compromised"


class DiscourseMarkerMetric:
    name = "Discourse Marker Density"

    DISCOURSE_MARKERS = {
        "addition": ["furthermore", "moreover", "additionally", "also", "besides", "likewise", "similarly"],
        "contrast": ["however", "nevertheless", "nonetheless", "conversely", "alternatively", "instead", "rather"],
        "causal": ["therefore", "thus", "consequently", "accordingly", "hence", "as a result", "so"],
        "temporal": ["meanwhile", "subsequently", "previously", "thereafter", "afterward", "eventually"],
        "emphasis": ["indeed", "in fact", "specifically", "particularly", "especially", "notably"],
        "clarification": ["in other words", "that is", "namely", "put differently", "to clarify"],
        "example": ["for example", "for instance", "e.g.", "i.e.", "such as", "namely"],
        "summary": ["in conclusion", "to summarize", "in summary", "overall", "all in all"],
    }

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None
        text_lower = stats.text.lower()
        marker_count = 0
        category_counts = {cat: 0 for cat in self.DISCOURSE_MARKERS}

        for category, markers in self.DISCOURSE_MARKERS.items():
            for marker in markers:
                count = text_lower.count(marker)
                marker_count += count
                category_counts[category] += count

        raw_value = (marker_count / max(stats.word_count, 1)) * 100

        std = standards.get("discourse_marker_density")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        if raw_value > 15:
            warning = "Very high discourse marker density suggests heavy AI editing"

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={"total_markers": marker_count, "category_breakdown": category_counts},
            warning=warning,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if ai_std <= human_std:
            return 0.5
        if value <= human_std:
            return 1.0
        if value >= ai_std:
            return 0.0
        return 1.0 - (value - human_std) / (ai_std - human_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.7:
            return "Preserved"
        if normalized >= 0.4:
            return "Moderate"
        return "Compromised"


class InformationDensityMetric:
    name = "Information Density"

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        content_count = sum(1 for word, tag in stats.pos_tags if tag in TextPreprocessor.CONTENT_POS)
        proper_nouns = sum(1 for word, tag in stats.pos_tags if tag in {"NNP", "NNPS"})

        content_ratio = content_count / max(stats.word_count, 1)
        proper_ratio = proper_nouns / max(stats.word_count, 1)
        raw_value = (content_ratio * 0.7) + (proper_ratio * 0.3)

        std = standards.get("information_density")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={"content_words": content_count, "proper_nouns": proper_nouns},
            warning=None,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if human_std <= ai_std:
            return 0.5
        if value >= human_std:
            return min(1.0, 0.5 + 0.5 * (value - human_std) / human_std)
        if value <= ai_std:
            return max(0.0, 0.5 * (value / ai_std))
        return (value - ai_std) / (human_std - ai_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.7:
            return "Preserved"
        if normalized >= 0.4:
            return "Moderate"
        return "Compromised"


class EpistemicHedgingMetric:
    name = "Epistemic Hedging"

    HEDGES = {
        "might",
        "may",
        "could",
        "would",
        "should",
        "can",
        "will",
        "suggests",
        "indicates",
        "implies",
        "appears",
        "seems",
        "tends",
        "believes",
        "assumes",
        "estimates",
        "speculates",
        "posits",
        "possibly",
        "probably",
        "likely",
        "perhaps",
        "maybe",
        "presumably",
        "apparently",
        "seemingly",
        "potentially",
        "arguably",
        "roughly",
        "possible",
        "probable",
        "unlikely",
        "potential",
        "tentative",
    }

    CERTAINTY_MARKERS = {
        "definitely",
        "certainly",
        "absolutely",
        "clearly",
        "obviously",
        "undoubtedly",
        "proves",
        "demonstrates",
        "shows",
        "confirms",
    }

    def calculate(self, stats: TextStats, standards: CalibrationStandards) -> MetricResult:
        warning = None
        text_lower = stats.text.lower()
        hedge_count = 0
        certainty_count = 0

        for hedge in self.HEDGES:
            hedge_count += text_lower.count(hedge)
        for certain in self.CERTAINTY_MARKERS:
            certainty_count += text_lower.count(certain)

        adjusted_hedge = max(0, hedge_count - (certainty_count * 0.5))
        raw_value = (adjusted_hedge / max(stats.word_count, 1)) * 100

        std = standards.get("epistemic_hedging")
        normalized = self._normalize(raw_value, std["human"], std["ai"])
        verdict = self._get_verdict(normalized)

        if raw_value < 0.02:
            warning = "Very low hedging suggests AI-like certainty"

        return MetricResult(
            name=self.name,
            raw_value=round(raw_value, 4),
            normalized_score=round(normalized, 4),
            human_standard=std["human"],
            ai_standard=std["ai"],
            verdict=verdict,
            details={
                "hedge_count": hedge_count,
                "certainty_count": certainty_count,
                "adjusted_hedge": round(adjusted_hedge, 2),
            },
            warning=warning,
        )

    def _normalize(self, value: float, human_std: float, ai_std: float) -> float:
        if human_std <= ai_std:
            return 0.5
        if value >= human_std:
            return min(1.0, 0.5 + 0.5 * (value - human_std) / human_std)
        if value <= ai_std:
            return max(0.0, 0.5 * (value / ai_std))
        return (value - ai_std) / (human_std - ai_std)

    def _get_verdict(self, normalized: float) -> str:
        if normalized >= 0.6:
            return "Preserved"
        if normalized >= 0.3:
            return "Moderate"
        return "Compromised"


class VoicePreservationScore:
    """Aggregates all 8 metrics into final Voice Preservation Score."""

    WEIGHTS = {
        "authenticity": 0.25,
        "lexical": 0.20,
        "structural": 0.20,
        "stylistic": 0.25,
        "consistency": 0.10,
    }

    AI_ISM_PENALTY_MAX = 30

    def __init__(self, standards: Optional[CalibrationStandards] = None):
        self.standards = standards or CalibrationStandards()
        self.preprocessor = TextPreprocessor()

        self.metrics = {
            "burstiness": BurstinessMetric(),
            "lexical_diversity": LexicalDiversityMetric(),
            "syntactic_complexity": SyntacticComplexityMetric(),
            "ai_ism_likelihood": AIIsmMetric(),
            "function_word_ratio": FunctionWordRatioMetric(),
            "discourse_marker_density": DiscourseMarkerMetric(),
            "information_density": InformationDensityMetric(),
            "epistemic_hedging": EpistemicHedgingMetric(),
        }

    def analyze(self, original_text: str, edited_text: str) -> Dict:
        original_stats = self.preprocessor.preprocess(original_text)
        edited_stats = self.preprocessor.preprocess(edited_text)

        results = {}
        for key, metric in self.metrics.items():
            results[key] = metric.calculate(edited_stats, self.standards)

        components = self._calculate_components(results)
        component_values = list(components.values())
        variance = self._calculate_variance(component_values)
        consistency_score = max(0, 1 - variance)

        voice_score = (
            components["authenticity"] * self.WEIGHTS["authenticity"]
            + components["lexical"] * self.WEIGHTS["lexical"]
            + components["structural"] * self.WEIGHTS["structural"]
            + components["stylistic"] * self.WEIGHTS["stylistic"]
            + consistency_score * self.WEIGHTS["consistency"]
        ) * 100

        ai_ism_result = results["ai_ism_likelihood"]
        penalty = self._calculate_ai_ism_penalty(ai_ism_result)
        final_score = max(0, voice_score - penalty)

        classification = self._classify(final_score)

        return {
            "original_stats": {
                "word_count": original_stats.word_count,
                "sentence_count": original_stats.sentence_count,
                "is_short_text": original_stats.is_short_text,
            },
            "edited_stats": {
                "word_count": edited_stats.word_count,
                "sentence_count": edited_stats.sentence_count,
                "is_short_text": edited_stats.is_short_text,
            },
            "metric_results": results,
            "component_scores": {k: round(v * 100, 2) for k, v in components.items()},
            "consistency_score": round(consistency_score * 100, 2),
            "voice_preservation_score": round(voice_score, 2),
            "ai_ism_penalty": round(penalty, 2),
            "final_score": round(final_score, 2),
            "classification": classification,
        }

    def _calculate_components(self, results: Dict[str, MetricResult]) -> Dict[str, float]:
        return {
            "authenticity": results["epistemic_hedging"].normalized_score,
            "lexical": results["lexical_diversity"].normalized_score,
            "structural": results["syntactic_complexity"].normalized_score,
            "stylistic": results["burstiness"].normalized_score,
        }

    def _calculate_variance(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return min(1.0, variance * 4)

    def _calculate_ai_ism_penalty(self, ai_ism_result: MetricResult) -> float:
        raw_value = ai_ism_result.raw_value
        std = self.standards.get("ai_ism_likelihood")
        excess = max(0, raw_value - std["human"])
        penalty = excess * 2
        return min(penalty, self.AI_ISM_PENALTY_MAX)

    def _classify(self, score: float) -> Dict:
        if score >= 80:
            return {"label": "Strong Voice Preserved", "color": "green", "level": 1}
        if score >= 60:
            return {"label": "Moderate Homogenization", "color": "yellow", "level": 2}
        if score >= 40:
            return {"label": "Significant Homogenization", "color": "orange", "level": 3}
        return {"label": "Severe Homogenization", "color": "red", "level": 4}


def analyze_texts(
    original_text: str,
    edited_text: str,
    custom_standards: Optional[Dict[str, Dict[str, float]]] = None,
) -> AnalysisResult:
    engine = VoicePreservationScore(
        CalibrationStandards(custom_standards) if custom_standards else None
    )
    result = engine.analyze(original_text, edited_text)

    edited_metric_results = result["metric_results"]
    original_metric_results = _calculate_metric_results(engine, original_text)

    metrics_edited = _format_metric_scores(edited_metric_results)
    metrics_original = _format_metric_scores(original_metric_results)
    components = _format_component_scores(result["component_scores"])
    classification = result["classification"]["label"]
    metric_standards = _format_metric_standards(engine.standards)

    original_stats = result["original_stats"]
    edited_stats = result["edited_stats"]

    edited_ai_ism_details = edited_metric_results["ai_ism_likelihood"].details
    ai_ism_categories = edited_ai_ism_details.get("category_breakdown", {})
    ai_ism_phrases = edited_ai_ism_details.get("detected_phrases", [])
    original_ai_ism_details = original_metric_results["ai_ism_likelihood"].details
    ai_ism_categories_original = original_ai_ism_details.get("category_breakdown", {})

    sentence_lengths = {
        "Original": _sentence_lengths(original_text),
        "Edited": _sentence_lengths(edited_text),
    }

    return AnalysisResult(
        score=result["final_score"],
        classification=classification,
        metrics_edited=metrics_edited,
        metrics_original=metrics_original,
        components=components,
        metric_standards=metric_standards,
        metric_results=edited_metric_results,
        consistency_score=result["consistency_score"],
        word_delta=edited_stats["word_count"] - original_stats["word_count"],
        sentence_delta=edited_stats["sentence_count"] - original_stats["sentence_count"],
        ai_ism_total=sum(ai_ism_categories.values()) if ai_ism_categories else 0,
        ai_ism_categories=ai_ism_categories,
        ai_ism_categories_original=ai_ism_categories_original,
        ai_ism_phrases=ai_ism_phrases,
        sentence_lengths=sentence_lengths,
        original_word_count=original_stats["word_count"],
        edited_word_count=edited_stats["word_count"],
        original_sentence_count=original_stats["sentence_count"],
        edited_sentence_count=edited_stats["sentence_count"],
    )


def _format_metric_scores(results: Dict[str, MetricResult]) -> Dict[str, float]:
    mapping = {
        "burstiness": "Burstiness",
        "lexical_diversity": "Lexical Diversity",
        "syntactic_complexity": "Syntactic Complexity",
        "ai_ism_likelihood": "AI-ism Likelihood",
        "function_word_ratio": "Function Word Ratio",
        "discourse_marker_density": "Discourse Markers",
        "information_density": "Information Density",
        "epistemic_hedging": "Epistemic Hedging",
    }
    formatted = {}
    for key, label in mapping.items():
        score = results[key].normalized_score * 100
        formatted[label] = round(score, 2)
    return formatted


def _format_component_scores(components: Dict[str, float]) -> Dict[str, float]:
    return {
        "Authenticity": components.get("authenticity", 0.0),
        "Lexical": components.get("lexical", 0.0),
        "Structural": components.get("structural", 0.0),
        "Stylistic": components.get("stylistic", 0.0),
    }


def _format_metric_standards(standards: CalibrationStandards) -> Dict[str, Dict[str, float]]:
    mapping = {
        "burstiness": "Burstiness",
        "lexical_diversity": "Lexical Diversity",
        "syntactic_complexity": "Syntactic Complexity",
        "ai_ism_likelihood": "AI-ism Likelihood",
        "function_word_ratio": "Function Word Ratio",
        "discourse_marker_density": "Discourse Markers",
        "information_density": "Information Density",
        "epistemic_hedging": "Epistemic Hedging",
    }
    formatted: Dict[str, Dict[str, float]] = {}
    for key, label in mapping.items():
        values = standards.get(key)
        formatted[label] = {
            "human": round(values["human"], 4),
            "ai": round(values["ai"], 4),
        }
    return formatted


def _calculate_metric_results(
    engine: "VoicePreservationScore",
    text: str,
) -> Dict[str, MetricResult]:
    stats = engine.preprocessor.preprocess(text)
    results: Dict[str, MetricResult] = {}
    for key, metric in engine.metrics.items():
        results[key] = metric.calculate(stats, engine.standards)
    return results


def _sentence_lengths(text: str) -> List[int]:
    preprocessor = TextPreprocessor()
    sentences = preprocessor._sentence_tokenize(text.strip()) if text.strip() else []
    return [len(preprocessor._word_tokenize(sentence)) for sentence in sentences] or [0]
