"""Analysis and metric calculation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import re
import statistics
from typing import Dict, List, Tuple


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
}

SUBORDINATORS = {
    "because",
    "since",
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
    "when",
    "where",
    "why",
    "how",
}

DISCOURSE_MARKERS = {
    "furthermore",
    "moreover",
    "additionally",
    "however",
    "therefore",
    "thus",
    "consequently",
    "in conclusion",
    "to summarize",
}

HEDGES = {
    "might",
    "may",
    "could",
    "would",
    "suggests",
    "indicates",
    "appears",
    "seems",
    "possibly",
    "probably",
}

AI_ISMS = {
    "connectors": ["moreover", "furthermore", "in addition", "as a result"],
    "cliches": ["it is important to note", "delve into", "plays a crucial role"],
    "generic_openers": ["in this essay", "this paper examines", "the purpose of this"],
}

METRIC_RANGES = {
    "Burstiness": (0.0, 0.8, False),
    "Lexical Diversity": (0.2, 0.8, False),
    "Syntactic Complexity": (0.1, 0.6, False),
    "AI-ism Likelihood": (0.0, 6.0, True),
    "Function Word Ratio": (0.35, 0.65, True),
    "Discourse Marker Density": (0.0, 8.0, True),
    "Information Density": (0.3, 0.7, False),
    "Epistemic Hedging": (0.0, 3.0, False),
}


@dataclass
class AnalysisResult:
    score: float
    classification: str
    metrics: Dict[str, float]
    components: Dict[str, float]
    word_delta: int
    sentence_delta: int
    ai_ism_total: int
    ai_ism_categories: Dict[str, int]
    sentence_lengths: Dict[str, List[int]]
    original_word_count: int
    edited_word_count: int
    original_sentence_count: int
    edited_sentence_count: int


def analyze_texts(original_text: str, edited_text: str) -> AnalysisResult:
    original_stats = _compute_stats(original_text)
    edited_stats = _compute_stats(edited_text)

    metrics = _score_metrics(edited_stats)
    components = {
        "Authenticity": metrics["Epistemic Hedging"],
        "Lexical": metrics["Lexical Diversity"],
        "Structural": metrics["Syntactic Complexity"],
        "Stylistic": metrics["Burstiness"],
    }

    score = round(sum(components.values()) / max(len(components), 1), 2)
    classification = _classify(score)

    return AnalysisResult(
        score=score,
        classification=classification,
        metrics=metrics,
        components=components,
        word_delta=edited_stats.word_count - original_stats.word_count,
        sentence_delta=edited_stats.sentence_count - original_stats.sentence_count,
        ai_ism_total=sum(edited_stats.ai_ism_categories.values()),
        ai_ism_categories=edited_stats.ai_ism_categories,
        sentence_lengths={
            "Original": original_stats.sentence_lengths,
            "Edited": edited_stats.sentence_lengths,
        },
        original_word_count=original_stats.word_count,
        edited_word_count=edited_stats.word_count,
        original_sentence_count=original_stats.sentence_count,
        edited_sentence_count=edited_stats.sentence_count,
    )


@dataclass
class _TextStats:
    words: List[str]
    word_count: int
    sentence_count: int
    sentence_lengths: List[int]
    subordination_ratio: float
    lexical_diversity: float
    burstiness: float
    function_word_ratio: float
    discourse_marker_density: float
    information_density: float
    epistemic_hedging: float
    ai_ism_likelihood: float
    ai_ism_categories: Dict[str, int]


def _tokenize_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _tokenize_words(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _compute_stats(text: str) -> _TextStats:
    sentences = _tokenize_sentences(text)
    words = _tokenize_words(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    sentence_lengths = [len(_tokenize_words(sentence)) for sentence in sentences] or [0]

    burstiness = _coefficient_of_variation(sentence_lengths)
    lexical_diversity = len(set(words)) / max(word_count, 1)
    subordination_ratio = _subordination_ratio(sentences)
    function_word_ratio = _ratio_in_set(words, FUNCTION_WORDS)
    discourse_marker_density = _density_of_phrases(text.lower(), DISCOURSE_MARKERS)
    information_density = _content_ratio(words)
    epistemic_hedging = _density_of_phrases(text.lower(), HEDGES)
    ai_ism_likelihood, ai_ism_categories = _ai_ism_density(text.lower())

    return _TextStats(
        words=words,
        word_count=word_count,
        sentence_count=sentence_count,
        sentence_lengths=sentence_lengths,
        subordination_ratio=subordination_ratio,
        lexical_diversity=lexical_diversity,
        burstiness=burstiness,
        function_word_ratio=function_word_ratio,
        discourse_marker_density=discourse_marker_density,
        information_density=information_density,
        epistemic_hedging=epistemic_hedging,
        ai_ism_likelihood=ai_ism_likelihood,
        ai_ism_categories=ai_ism_categories,
    )


def _score_metrics(stats: _TextStats) -> Dict[str, float]:
    raw_metrics = {
        "Burstiness": stats.burstiness,
        "Lexical Diversity": stats.lexical_diversity,
        "Syntactic Complexity": stats.subordination_ratio,
        "AI-ism Likelihood": stats.ai_ism_likelihood,
        "Function Word Ratio": stats.function_word_ratio,
        "Discourse Marker Density": stats.discourse_marker_density,
        "Information Density": stats.information_density,
        "Epistemic Hedging": stats.epistemic_hedging,
    }

    scored = {}
    for name, value in raw_metrics.items():
        min_val, max_val, invert = METRIC_RANGES[name]
        scored[name] = round(_scale_metric(value, min_val, max_val, invert), 2)
    return scored


def _scale_metric(value: float, min_val: float, max_val: float, invert: bool) -> float:
    if max_val <= min_val:
        return 0.0
    normalized = (value - min_val) / (max_val - min_val)
    normalized = min(max(normalized, 0.0), 1.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized * 100


def _coefficient_of_variation(values: List[int]) -> float:
    if len(values) < 2:
        return 0.0
    mean_val = sum(values) / len(values)
    if mean_val <= 0:
        return 0.0
    return statistics.pstdev(values) / mean_val


def _subordination_ratio(sentences: List[str]) -> float:
    if not sentences:
        return 0.0
    count = 0
    for sentence in sentences:
        words = sentence.lower().split()
        if any(word in SUBORDINATORS for word in words):
            count += 1
    return count / max(len(sentences), 1)


def _ratio_in_set(words: List[str], target: set[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for word in words if word in target) / len(words)


def _density_of_phrases(text: str, phrases: set[str]) -> float:
    if not text.strip():
        return 0.0
    count = sum(text.count(phrase) for phrase in phrases)
    word_count = max(len(_tokenize_words(text)), 1)
    return (count / word_count) * 100


def _content_ratio(words: List[str]) -> float:
    if not words:
        return 0.0
    content_words = [word for word in words if word not in FUNCTION_WORDS]
    return len(content_words) / len(words)


def _ai_ism_density(text: str) -> Tuple[float, Dict[str, int]]:
    word_count = max(len(_tokenize_words(text)), 1)
    categories = {category: 0 for category in AI_ISMS}
    total = 0
    for category, phrases in AI_ISMS.items():
        for phrase in phrases:
            count = text.count(phrase)
            categories[category] += count
            total += count
    density = (total / word_count) * 100
    return density, categories


def _classify(score: float) -> str:
    if score >= 80:
        return "Strong Voice Preserved"
    if score >= 60:
        return "Moderate Homogenization"
    if score >= 40:
        return "Significant Homogenization"
    return "Severe Homogenization"
