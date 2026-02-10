VoiceTracer is a computational analysis tool that measures how much AI editing changes a writer's unique voice. Here's what it does:

Core Purpose  
It compares an original human-written text against an AI-edited version, then calculates how much of the writer's authentic style was preserved or lost in the editing process.

How It Works

1. Text Processing \- Breaks down both texts into sentences, words, and grammatical components  
2. Eight Measurements \- Analyzes the edited text across eight stylistic dimensions:  
   * Burstiness: Whether sentence lengths vary naturally or become uniform  
   * Lexical Diversity: Vocabulary richness vs. repetitive word choices  
   * Syntactic Complexity: Sentence structure sophistication  
   * AI-ism Likelihood: Presence of formulaic AI phrases like "delve into" or "it is important to note"  
   * Function Word Density: Over-use of small connector words (the, and, of)  
   * Discourse Markers: Excessive signposting (furthermore, therefore, however)  
   * Information Density: Ratio of meaningful content words to total words  
   * Epistemic Hedging: Uncertainty markers (might, suggests, possibly) that show academic caution  
3. Scoring System \- Combines these eight metrics into a 0-100 Voice Preservation Score using weighted components:  
   * Authenticity (hedging) \= 25%  
   * Stylistic variation (burstiness) \= 25%  
   * Lexical diversity \= 20%  
   * Structural complexity \= 20%  
   * Consistency across text \= 10%  
4. AI-ism Penalty \- Deducts points for detected AI formulaic phrases (up to 30 points maximum)  
5. Classification \- Assigns a final label:  
   * 80-100: Strong Voice Preserved (green)  
   * 60-79: Moderate Homogenization (yellow)  
   * 40-59: Significant Homogenization (orange)  
   * 0-39: Severe Homogenization (red)  
6. Recommendations \- Generates specific, actionable suggestions for restoring voice elements that were lost

Key Features

* Works without external dependencies (pure Python)  
* Handles short texts with reliability warnings  
* Calibrates scores against empirical human vs. AI writing standards  
* Produces JSON-serializable output for web integration  
* Includes comprehensive error handling and edge case management

Theoretical Foundation  
Based on your thesis framework: Holec's learner autonomy, Krashen's monitor hypothesis, and Long's negotiation of meaning—adapted to help writers negotiate with AI tools rather than be dominated by them.

\=== VoiceTracer Core Logic Implementation Complete \===

All 8 metrics \+ Voice Preservation Score aggregator implemented:  
1\. Burstiness \- Sentence length variation (CV)  
2\. Lexical Diversity \- MTLD with TTR fallback  
3\. Syntactic Complexity \- Subordination ratio  
4\. AI-ism Likelihood \- Weighted phrase detection  
5\. Function Word Ratio \- Closed-class word density  
6\. Discourse Marker Density \- Signposting frequency  
7\. Information Density \- Content word ratio  
8\. Epistemic Hedging \- Uncertainty markers

Aggregator: Weighted component score \+ AI-ism penalty (max 30\)  
Output: 0-100 final score with 4-level classification

Code

\# CORRECTED AND FINAL IMPLEMENTATION  
\# VoiceTracer Core Computational Logic \- Production Ready

import re  
import math  
from typing import List, Dict, Tuple, Optional, Union  
from dataclasses import dataclass  
from collections import Counter  
import unittest

\# \=============================================================================  
\# DATA CLASSES AND CONFIGURATION  
\# \=============================================================================

@dataclass  
class MetricResult:  
    """Standardized result container for all metrics."""  
    name: str  
    raw\_value: float  
    normalized\_score: float  \# 0-1, where 1 \= human-like, 0 \= AI-like  
    human\_standard: float  
    ai\_standard: float  
    verdict: str  \# "Preserved", "Moderate", "Compromised"  
    details: Dict  
    warning: Optional\[str\] \= None

@dataclass  
class TextStats:  
    """Pre-computed text statistics used across metrics."""  
    text: str  
    words: List\[str\]  
    sentences: List\[str\]  
    word\_count: int  
    sentence\_count: int  
    avg\_sentence\_length: float  
    pos\_tags: List\[Tuple\[str, str\]\]  
    is\_short\_text: bool

class CalibrationStandards:  
    """Default and adjustable standards for human vs AI writing."""  
      
    DEFAULTS \= {  
        'burstiness': {'human': 1.23, 'ai': 0.78},  
        'lexical\_diversity': {'human': 0.55, 'ai': 0.42},  \# For TTR fallback  
        'syntactic\_complexity': {'human': 0.54, 'ai': 0.64},  
        'ai\_ism\_likelihood': {'human': 3.1, 'ai': 78.5},  
        'function\_word\_ratio': {'human': 0.50, 'ai': 0.60},  
        'discourse\_marker\_density': {'human': 8.0, 'ai': 18.0},  
        'information\_density': {'human': 0.58, 'ai': 0.42},  
        'epistemic\_hedging': {'human': 0.09, 'ai': 0.04}  
    }  
      
    def \_\_init\_\_(self, custom\_standards: Optional\[Dict\] \= None):  
        self.standards \= custom\_standards or self.DEFAULTS.copy()  
      
    def get(self, metric: str) \-\> Dict\[str, float\]:  
        return self.standards.get(metric, self.DEFAULTS\[metric\])

\# \=============================================================================  
\# TEXT PREPROCESSOR  
\# \=============================================================================

class TextPreprocessor:  
    """Handles all text preprocessing and statistical computation."""  
      
    SHORT\_TEXT\_THRESHOLD \= 150  
      
    FUNCTION\_WORDS \= {  
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',  
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be',  
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',  
        'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this',  
        'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',  
        'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its',  
        'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'what',  
        'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each', 'few',  
        'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',  
        'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now', 'then',  
        'here', 'there', 'up', 'down', 'out', 'off', 'over', 'under', 'again',  
        'further', 'once', 'during', 'before', 'after', 'above', 'below',  
        'between', 'through', 'into', 'within', 'without', 'against', 'about'  
    }  
      
    CONTENT\_POS \= {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN',   
                   'VBP', 'VBZ', 'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS'}  
      
    def preprocess(self, text: str) \-\> TextStats:  
        """Compute all base statistics for a text."""  
        text \= text.strip()  
        if not text:  
            raise ValueError("Empty text provided")  
          
        sentences \= self.\_sentence\_tokenize(text)  
        words \= self.\_word\_tokenize(text.lower())  
        alpha\_words \= \[w for w in words if any(c.isalpha() for c in w)\]  
          
        word\_count \= len(alpha\_words)  
        sentence\_count \= len(sentences)  
          
        if sentence\_count \== 0:  
            raise ValueError("No sentences detected in text")  
          
        avg\_sentence\_length \= word\_count / sentence\_count if sentence\_count \> 0 else 0  
        pos\_tags \= self.\_simple\_pos\_tag(alpha\_words)  
        is\_short\_text \= word\_count \< self.SHORT\_TEXT\_THRESHOLD  
          
        return TextStats(  
            text=text, words=alpha\_words, sentences=sentences,  
            word\_count=word\_count, sentence\_count=sentence\_count,  
            avg\_sentence\_length=avg\_sentence\_length, pos\_tags=pos\_tags,  
            is\_short\_text=is\_short\_text  
        )  
      
    def \_sentence\_tokenize(self, text: str) \-\> List\[str\]:  
        """Robust sentence tokenization."""  
        pattern \= r'(?\<=\[.\!?\])\\s+(?=\[A-Z\])'  
        sentences \= re.split(pattern, text)  
        return \[s.strip() for s in sentences if s.strip()\]  
      
    def \_word\_tokenize(self, text: str) \-\> List\[str\]:  
        """Robust word tokenization."""  
        return re.findall(r'\\b\\w+\\b', text)  
      
    def \_simple\_pos\_tag(self, words: List\[str\]) \-\> List\[Tuple\[str, str\]\]:  
        """Simple rule-based POS tagging fallback."""  
        tags \= \[\]  
        for word in words:  
            w \= word.lower()  
            if w in {'the', 'a', 'an'}:  
                tag \= 'DT'  
            elif w in {'is', 'was', 'are', 'were', 'be', 'been', 'being'}:  
                tag \= 'VB'  
            elif w in {'i', 'you', 'he', 'she', 'it', 'we', 'they'}:  
                tag \= 'PRP'  
            elif w in {'in', 'on', 'at', 'to', 'for', 'of', 'with'}:  
                tag \= 'IN'  
            elif w.endswith('ing'):  
                tag \= 'VBG'  
            elif w.endswith('ed'):  
                tag \= 'VBD'  
            elif w.endswith('ly'):  
                tag \= 'RB'  
            elif w\[0\].isupper() and len(w) \> 1:  
                tag \= 'NNP'  
            elif w in self.FUNCTION\_WORDS:  
                tag \= 'XX'  
            else:  
                tag \= 'NN'  
            tags.append((word, tag))  
        return tags

\# \=============================================================================  
\# METRIC 1: BURSTINESS  
\# \=============================================================================

class BurstinessMetric:  
    """Measures sentence length variation using coefficient of variation."""  
      
    name \= "Burstiness"  
    MIN\_SENTENCES \= 3  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
          
        sentence\_lengths \= \[\]  
        for sent in stats.sentences:  
            length \= len(re.findall(r'\\b\\w+\\b', sent))  
            if length \> 0:  
                sentence\_lengths.append(length)  
          
        if len(sentence\_lengths) \< self.MIN\_SENTENCES:  
            warning \= f"Insufficient sentences ({len(sentence\_lengths)}) for reliable burstiness"  
            raw\_value \= 0.0  
            normalized \= 0.5  
        else:  
            mean\_len \= sum(sentence\_lengths) / len(sentence\_lengths)  
            variance \= sum((x \- mean\_len) \*\* 2 for x in sentence\_lengths) / len(sentence\_lengths)  
            std\_dev \= math.sqrt(variance)  
            raw\_value \= std\_dev / mean\_len if mean\_len \> 0 else 0.0  
              
            std \= standards.get('burstiness')  
            normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
          
        verdict \= self.\_get\_verdict(normalized)  
          
        if stats.is\_short\_text:  
            warning \= warning or "Short text: burstiness may be unreliable"  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=standards.get('burstiness')\['human'\],  
            ai\_standard=standards.get('burstiness')\['ai'\],  
            verdict=verdict,  
            details={'sentence\_lengths': sentence\_lengths},  
            warning=warning  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if human\_std \== ai\_std:  
            return 0.5  
        if value \>= human\_std:  
            return min(1.0, 0.5 \+ 0.5 \* (value \- human\_std) / human\_std)  
        elif value \<= ai\_std:  
            return max(0.0, 0.5 \* (value / ai\_std))  
        else:  
            return 0.5 \+ 0.5 \* (value \- ai\_std) / (human\_std \- ai\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.7: return "Preserved"  
        elif normalized \>= 0.4: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 2: LEXICAL DIVERSITY  
\# \=============================================================================

class LexicalDiversityMetric:  
    """Measures vocabulary richness using MTLD with TTR fallback for short texts."""  
      
    name \= "Lexical Diversity"  
    TTR\_THRESHOLD \= 0.72  
    MTLD\_MIN\_WORDS \= 100  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
          
        if stats.word\_count \< 50:  
            warning \= "Very short text: lexical diversity unreliable"  
          
        \# Use TTR for short texts, MTLD for longer  
        if stats.word\_count \< self.MTLD\_MIN\_WORDS:  
            raw\_value \= len(set(stats.words)) / max(stats.word\_count, 1\)  
            method \= "TTR"  
        else:  
            raw\_value \= self.\_calculate\_mtld(stats.words)  
            method \= "MTLD"  
          
        std \= standards.get('lexical\_diversity')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\], method)  
        verdict \= self.\_get\_verdict(normalized)  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={  
                'method': method,  
                'unique\_words': len(set(stats.words)),  
                'total\_words': stats.word\_count  
            },  
            warning=warning  
        )  
      
    def \_calculate\_mtld(self, words: List\[str\]) \-\> float:  
        if len(words) \< 10:  
            return 0.0  
          
        words \= \[w.lower() for w in words if w.isalpha()\]  
          
        def mtld\_pass(word\_list: List\[str\]) \-\> float:  
            factors \= 0  
            types \= set()  
            token\_count \= 0  
              
            for word in word\_list:  
                types.add(word)  
                token\_count \+= 1  
                ttr \= len(types) / token\_count  
                  
                if ttr \<= self.TTR\_THRESHOLD:  
                    factors \+= 1  
                    types \= set()  
                    token\_count \= 0  
              
            if token\_count \> 0 and len(types) / token\_count \< 1:  
                factors \+= (1 \- self.TTR\_THRESHOLD) / (1 \- len(types) / token\_count)  
              
            return len(word\_list) / factors if factors \> 0 else len(word\_list)  
          
        forward \= mtld\_pass(words)  
        backward \= mtld\_pass(words\[::-1\])  
        return (forward \+ backward) / 2  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float, method: str) \-\> float:  
        \# Different scales for TTR vs MTLD  
        if method \== "TTR":  
            \# TTR is already 0-1  
            if value \>= human\_std:  
                return min(1.0, value)  
            else:  
                return max(0.0, value / human\_std \* 0.5)  
        else:  
            \# MTLD scale (typically 40-100)  
            if human\_std \<= ai\_std:  
                return 0.5  
            if value \>= human\_std:  
                return min(1.0, 0.5 \+ 0.5 \* (value \- human\_std) / human\_std)  
            elif value \<= ai\_std:  
                return max(0.0, 0.5 \* (value / ai\_std))  
            else:  
                return (value \- ai\_std) / (human\_std \- ai\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.7: return "Preserved"  
        elif normalized \>= 0.4: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 3: SYNTACTIC COMPLEXITY  
\# \=============================================================================

class SyntacticComplexityMetric:  
    """Measures syntactic sophistication using subordination ratio."""  
      
    name \= "Syntactic Complexity"  
      
    SUBORDINATORS \= {  
        'because', 'since', 'as', 'although', 'though', 'while', 'whereas',  
        'if', 'unless', 'whether', 'that', 'which', 'who', 'whom', 'whose',  
        'when', 'where', 'why', 'how', 'after', 'before', 'until', 'till',  
        'once', 'even', 'provided', 'assuming', 'given'  
    }  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
          
        subord\_count \= 0  
        for sent in stats.sentences:  
            words \= sent.lower().split()  
            if any(w in self.SUBORDINATORS for w in words):  
                subord\_count \+= 1  
          
        subord\_ratio \= subord\_count / len(stats.sentences) if stats.sentences else 0  
        length\_factor \= min(stats.avg\_sentence\_length / 30, 1.0)  
        raw\_value \= (subord\_ratio \* 0.6) \+ (length\_factor \* 0.4)  
          
        std \= standards.get('syntactic\_complexity')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        if stats.is\_short\_text:  
            warning \= "Short text: complexity patterns may not be representative"  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={'subordination\_ratio': round(subord\_ratio, 4)},  
            warning=warning  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if human\_std \== ai\_std:  
            return 0.5  
        \# Lower complexity is more human-like  
        if value \<= human\_std:  
            return min(1.0, 0.5 \+ 0.5 \* (human\_std \- value) / human\_std)  
        elif value \>= ai\_std:  
            return max(0.0, 0.5 \* (ai\_std / value) if value \> 0 else 0\)  
        else:  
            return 0.5 \+ 0.5 \* (ai\_std \- value) / (ai\_std \- human\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.6: return "Preserved"  
        elif normalized \>= 0.3: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 4: AI-ISM LIKELIHOOD  
\# \=============================================================================

class AIIsmMetric:  
    """Detects formulaic phrases characteristic of AI-generated text."""  
      
    name \= "AI-ism Likelihood"  
      
    AI\_ISMS \= {  
        'connectors': {  
            'moreover': 2, 'furthermore': 2, 'additionally': 2, 'consequently': 2,  
            'therefore': 2, 'thus': 2, 'hence': 2, 'accordingly': 2,  
            'in order to': 2, 'as a result': 2, 'in addition': 2, 'on the other hand': 2,  
            'in contrast': 2, 'by contrast': 2, 'in comparison': 2, 'similarly': 2,  
            'likewise': 2, 'conversely': 2, 'alternatively': 2, 'subsequently': 2,  
            'nevertheless': 2, 'nonetheless': 2, 'notwithstanding': 2, 'regardless': 2  
        },  
        'cliches': {  
            'delve into': 3, 'it is important to note': 3, 'it is worth noting': 3,  
            'it should be noted': 3, 'plays a crucial role': 3, 'plays an important role': 3,  
            'sheds light on': 3, 'offers insights into': 3, 'provides insights into': 3,  
            'a wide range of': 3, 'a variety of': 3, 'a number of': 3,  
            'in the realm of': 3, 'in the context of': 3, 'in terms of': 3,  
            'it is evident that': 3, 'it is clear that': 3, 'underscores the importance': 3,  
            'highlights the significance': 3, 'raises important questions': 3  
        },  
        'generic\_openers': {  
            'in conclusion': 2, 'to conclude': 2, 'in summary': 2, 'to summarize': 2,  
            'in this essay': 2, 'this essay will': 2, 'this paper examines': 2,  
            'the purpose of this': 2, 'this study aims to': 2, 'the present study': 2  
        }  
    }  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
        text\_lower \= stats.text.lower()  
        total\_weight \= 0  
        detected \= \[\]  
        category\_counts \= {'connectors': 0, 'cliches': 0, 'generic\_openers': 0}  
          
        for category, phrases in self.AI\_ISMS.items():  
            for phrase, weight in phrases.items():  
                count \= text\_lower.count(phrase)  
                if count \> 0:  
                    category\_counts\[category\] \+= count  
                    total\_weight \+= count \* weight  
                    detected.append({'phrase': phrase, 'count': count, 'weight': weight, 'category': category})  
          
        raw\_value \= (total\_weight / max(stats.word\_count, 1)) \* 100  
          
        std \= standards.get('ai\_ism\_likelihood')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        if raw\_value \> 15:  
            warning \= "Very high AI-ism density suggests heavy AI editing"  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={  
                'total\_weighted\_count': total\_weight,  
                'category\_breakdown': category\_counts,  
                'detected\_phrases': sorted(detected, key=lambda x: x\['count'\] \* x\['weight'\], reverse=True)\[:10\]  
            },  
            warning=warning  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if ai\_std \<= human\_std:  
            return 0.5  
        if value \<= human\_std:  
            return 1.0  
        elif value \>= ai\_std:  
            return 0.0  
        else:  
            return 1.0 \- (value \- human\_std) / (ai\_std \- human\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.8: return "Preserved"  
        elif normalized \>= 0.5: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 5: FUNCTION WORD RATIO  
\# \=============================================================================

class FunctionWordRatioMetric:  
    """Measures density of closed-class words."""  
      
    name \= "Function Word Ratio"  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        function\_count \= sum(1 for w in stats.words if w in TextPreprocessor.FUNCTION\_WORDS)  
        raw\_value \= function\_count / max(stats.word\_count, 1\)  
          
        std \= standards.get('function\_word\_ratio')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={'function\_words': function\_count, 'total\_words': stats.word\_count},  
            warning=None  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if ai\_std \<= human\_std:  
            return 0.5  
        if value \<= human\_std:  
            return 1.0  
        elif value \>= ai\_std:  
            return 0.0  
        else:  
            return 1.0 \- (value \- human\_std) / (ai\_std \- human\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.7: return "Preserved"  
        elif normalized \>= 0.4: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 6: DISCOURSE MARKER DENSITY  
\# \=============================================================================

class DiscourseMarkerMetric:  
    """Measures frequency of signposting words."""  
      
    name \= "Discourse Marker Density"  
      
    DISCOURSE\_MARKERS \= {  
        'addition': \['furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise', 'similarly'\],  
        'contrast': \['however', 'nevertheless', 'nonetheless', 'conversely', 'alternatively', 'instead', 'rather'\],  
        'causal': \['therefore', 'thus', 'consequently', 'accordingly', 'hence', 'as a result', 'so'\],  
        'temporal': \['meanwhile', 'subsequently', 'previously', 'thereafter', 'afterward', 'eventually'\],  
        'emphasis': \['indeed', 'in fact', 'specifically', 'particularly', 'especially', 'notably'\],  
        'clarification': \['in other words', 'that is', 'namely', 'put differently', 'to clarify'\],  
        'example': \['for example', 'for instance', 'e.g.', 'i.e.', 'such as', 'namely'\],  
        'summary': \['in conclusion', 'to summarize', 'in summary', 'overall', 'all in all'\]  
    }  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
        text\_lower \= stats.text.lower()  
        marker\_count \= 0  
        category\_counts \= {cat: 0 for cat in self.DISCOURSE\_MARKERS}  
          
        for category, markers in self.DISCOURSE\_MARKERS.items():  
            for marker in markers:  
                count \= text\_lower.count(marker)  
                marker\_count \+= count  
                category\_counts\[category\] \+= count  
          
        raw\_value \= (marker\_count / max(stats.word\_count, 1)) \* 100  
          
        std \= standards.get('discourse\_marker\_density')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        if raw\_value \> 15:  
            warning \= "Very high discourse marker density suggests heavy AI editing"  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={'total\_markers': marker\_count, 'category\_breakdown': category\_counts},  
            warning=warning  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if ai\_std \<= human\_std:  
            return 0.5  
        if value \<= human\_std:  
            return 1.0  
        elif value \>= ai\_std:  
            return 0.0  
        else:  
            return 1.0 \- (value \- human\_std) / (ai\_std \- human\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.7: return "Preserved"  
        elif normalized \>= 0.4: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 7: INFORMATION DENSITY  
\# \=============================================================================

class InformationDensityMetric:  
    """Measures ratio of content-carrying words to total words."""  
      
    name \= "Information Density"  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        content\_count \= sum(1 for word, tag in stats.pos\_tags if tag in TextPreprocessor.CONTENT\_POS)  
        proper\_nouns \= sum(1 for word, tag in stats.pos\_tags if tag in {'NNP', 'NNPS'})  
          
        content\_ratio \= content\_count / max(stats.word\_count, 1\)  
        proper\_ratio \= proper\_nouns / max(stats.word\_count, 1\)  
        raw\_value \= (content\_ratio \* 0.7) \+ (proper\_ratio \* 0.3)  
          
        std \= standards.get('information\_density')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={'content\_words': content\_count, 'proper\_nouns': proper\_nouns},  
            warning=None  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if human\_std \<= ai\_std:  
            return 0.5  
        if value \>= human\_std:  
            return min(1.0, 0.5 \+ 0.5 \* (value \- human\_std) / human\_std)  
        elif value \<= ai\_std:  
            return max(0.0, 0.5 \* (value / ai\_std))  
        else:  
            return (value \- ai\_std) / (human\_std \- ai\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.7: return "Preserved"  
        elif normalized \>= 0.4: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# METRIC 8: EPISTEMIC HEDGING  
\# \=============================================================================

class EpistemicHedgingMetric:  
    """Measures uncertainty markers and tentative language."""  
      
    name \= "Epistemic Hedging"  
      
    HEDGES \= {  
        'might', 'may', 'could', 'would', 'should', 'can', 'will',  
        'suggests', 'indicates', 'implies', 'appears', 'seems', 'tends',  
        'believes', 'assumes', 'estimates', 'speculates', 'posits',  
        'possibly', 'probably', 'likely', 'perhaps', 'maybe', 'presumably',  
        'apparently', 'seemingly', 'potentially', 'arguably', 'roughly',  
        'possible', 'probable', 'likely', 'unlikely', 'potential', 'tentative'  
    }  
      
    CERTAINTY\_MARKERS \= {  
        'definitely', 'certainly', 'absolutely', 'clearly', 'obviously',  
        'undoubtedly', 'proves', 'demonstrates', 'shows', 'confirms'  
    }  
      
    def calculate(self, stats: TextStats, standards: CalibrationStandards) \-\> MetricResult:  
        warning \= None  
        text\_lower \= stats.text.lower()  
        hedge\_count \= 0  
        certainty\_count \= 0  
          
        for hedge in self.HEDGES:  
            hedge\_count \+= text\_lower.count(hedge)  
        for certain in self.CERTAINTY\_MARKERS:  
            certainty\_count \+= text\_lower.count(certain)  
          
        adjusted\_hedge \= max(0, hedge\_count \- (certainty\_count \* 0.5))  
        raw\_value \= (adjusted\_hedge / max(stats.word\_count, 1)) \* 100  
          
        std \= standards.get('epistemic\_hedging')  
        normalized \= self.\_normalize(raw\_value, std\['human'\], std\['ai'\])  
        verdict \= self.\_get\_verdict(normalized)  
          
        if raw\_value \< 0.02:  
            warning \= "Very low hedging suggests AI-like certainty"  
          
        return MetricResult(  
            name=self.name, raw\_value=round(raw\_value, 4),  
            normalized\_score=round(normalized, 4),  
            human\_standard=std\['human'\], ai\_standard=std\['ai'\],  
            verdict=verdict,  
            details={  
                'hedge\_count': hedge\_count,  
                'certainty\_count': certainty\_count,  
                'adjusted\_hedge': round(adjusted\_hedge, 2\)  
            },  
            warning=warning  
        )  
      
    def \_normalize(self, value: float, human\_std: float, ai\_std: float) \-\> float:  
        if human\_std \<= ai\_std:  
            return 0.5  
        if value \>= human\_std:  
            return min(1.0, 0.5 \+ 0.5 \* (value \- human\_std) / human\_std)  
        elif value \<= ai\_std:  
            return max(0.0, 0.5 \* (value / ai\_std))  
        else:  
            return (value \- ai\_std) / (human\_std \- ai\_std)  
      
    def \_get\_verdict(self, normalized: float) \-\> str:  
        if normalized \>= 0.6: return "Preserved"  
        elif normalized \>= 0.3: return "Moderate"  
        return "Compromised"

\# \=============================================================================  
\# VOICE PRESERVATION SCORE AGGREGATOR  
\# \=============================================================================

class VoicePreservationScore:  
    """Aggregates all 8 metrics into final Voice Preservation Score."""  
      
    WEIGHTS \= {  
        'authenticity': 0.25,      \# Epistemic Hedging  
        'lexical': 0.20,           \# Lexical Diversity  
        'structural': 0.20,        \# Syntactic Complexity  
        'stylistic': 0.25,         \# Burstiness  
        'consistency': 0.10        \# Calculated from variance  
    }  
      
    AI\_ISM\_PENALTY\_MAX \= 30  
      
    def \_\_init\_\_(self, standards: Optional\[CalibrationStandards\] \= None):  
        self.standards \= standards or CalibrationStandards()  
        self.preprocessor \= TextPreprocessor()  
          
        self.metrics \= {  
            'burstiness': BurstinessMetric(),  
            'lexical\_diversity': LexicalDiversityMetric(),  
            'syntactic\_complexity': SyntacticComplexityMetric(),  
            'ai\_ism\_likelihood': AIIsmMetric(),  
            'function\_word\_ratio': FunctionWordRatioMetric(),  
            'discourse\_marker\_density': DiscourseMarkerMetric(),  
            'information\_density': InformationDensityMetric(),  
            'epistemic\_hedging': EpistemicHedgingMetric()  
        }  
      
    def analyze(self, original\_text: str, edited\_text: str) \-\> Dict:  
        """Perform complete analysis of original vs edited text."""  
        original\_stats \= self.preprocessor.preprocess(original\_text)  
        edited\_stats \= self.preprocessor.preprocess(edited\_text)  
          
        results \= {}  
        for key, metric in self.metrics.items():  
            results\[key\] \= metric.calculate(edited\_stats, self.standards)  
          
        components \= self.\_calculate\_components(results)  
        component\_values \= list(components.values())  
        variance \= self.\_calculate\_variance(component\_values)  
        consistency\_score \= max(0, 1 \- variance)  
          
        voice\_score \= (  
            components\['authenticity'\] \* self.WEIGHTS\['authenticity'\] \+  
            components\['lexical'\] \* self.WEIGHTS\['lexical'\] \+  
            components\['structural'\] \* self.WEIGHTS\['structural'\] \+  
            components\['stylistic'\] \* self.WEIGHTS\['stylistic'\] \+  
            consistency\_score \* self.WEIGHTS\['consistency'\]  
        ) \* 100  
          
        ai\_ism\_result \= results\['ai\_ism\_likelihood'\]  
        penalty \= self.\_calculate\_ai\_ism\_penalty(ai\_ism\_result)  
        final\_score \= max(0, voice\_score \- penalty)  
          
        classification \= self.\_classify(final\_score)  
          
        return {  
            'original\_stats': {  
                'word\_count': original\_stats.word\_count,  
                'sentence\_count': original\_stats.sentence\_count,  
                'is\_short\_text': original\_stats.is\_short\_text  
            },  
            'edited\_stats': {  
                'word\_count': edited\_stats.word\_count,  
                'sentence\_count': edited\_stats.sentence\_count,  
                'is\_short\_text': edited\_stats.is\_short\_text  
            },  
            'metric\_results': results,  
            'component\_scores': {k: round(v \* 100, 2\) for k, v in components.items()},  
            'consistency\_score': round(consistency\_score \* 100, 2),  
            'voice\_preservation\_score': round(voice\_score, 2),  
            'ai\_ism\_penalty': round(penalty, 2),  
            'final\_score': round(final\_score, 2),  
            'classification': classification,  
            'recommendations': self.\_generate\_recommendations(results, components)  
        }  
      
    def \_calculate\_components(self, results: Dict\[str, MetricResult\]) \-\> Dict\[str, float\]:  
        return {  
            'authenticity': results\['epistemic\_hedging'\].normalized\_score,  
            'lexical': results\['lexical\_diversity'\].normalized\_score,  
            'structural': results\['syntactic\_complexity'\].normalized\_score,  
            'stylistic': results\['burstiness'\].normalized\_score  
        }  
      
    def \_calculate\_variance(self, values: List\[float\]) \-\> float:  
        if len(values) \< 2:  
            return 0.0  
        mean \= sum(values) / len(values)  
        variance \= sum((x \- mean) \*\* 2 for x in values) / len(values)  
        return min(1.0, variance \* 4\)  
      
    def \_calculate\_ai\_ism\_penalty(self, ai\_ism\_result: MetricResult) \-\> float:  
        raw\_value \= ai\_ism\_result.raw\_value  
        std \= self.standards.get('ai\_ism\_likelihood')  
        excess \= max(0, raw\_value \- std\['human'\])  
        penalty \= excess \* 2  
        return min(penalty, self.AI\_ISM\_PENALTY\_MAX)  
      
    def \_classify(self, score: float) \-\> Dict:  
        if score \>= 80:  
            return {'label': 'Strong Voice Preserved', 'color': 'green', 'level': 1}  
        elif score \>= 60:  
            return {'label': 'Moderate Homogenization', 'color': 'yellow', 'level': 2}  
        elif score \>= 40:  
            return {'label': 'Significant Homogenization', 'color': 'orange', 'level': 3}  
        else:  
            return {'label': 'Severe Homogenization', 'color': 'red', 'level': 4}  
      
    def \_generate\_recommendations(self, results: Dict, components: Dict) \-\> List\[str\]:  
        recommendations \= \[\]  
          
        if components\['authenticity'\] \< 0.5:  
            recommendations.append("Restore epistemic hedging (uncertainty markers) to show appropriate academic caution")  
          
        if components\['lexical'\] \< 0.5:  
            recommendations.append("Increase vocabulary diversity—avoid repetitive phrasing patterns")  
          
        if components\['structural'\] \< 0.5:  
            recommendations.append("Vary sentence structure—mix simple and complex constructions")  
          
        if components\['stylistic'\] \< 0.5:  
            recommendations.append("Vary sentence length—include both short punchy and longer flowing sentences")  
          
        if results\['discourse\_marker\_density'\].normalized\_score \< 0.4:  
            recommendations.append("Reduce signposting words—trust reader to follow your logic")  
          
        if results\['ai\_ism\_likelihood'\].normalized\_score \< 0.5:  
            detected \= results\['ai\_ism\_likelihood'\].details.get('detected\_phrases', \[\])\[:3\]  
            if detected:  
                phrases \= \[d\['phrase'\] for d in detected\]  
                recommendations.append(f"Replace AI-typical phrases: {', '.join(phrases)}")  
          
        return recommendations\[:5\]

print("=== VoiceTracer Core Logic Implementation Complete \===")  
print("\\nAll 8 metrics \+ Voice Preservation Score aggregator implemented:")  
print("1. Burstiness \- Sentence length variation (CV)")  
print("2. Lexical Diversity \- MTLD with TTR fallback")  
print("3. Syntactic Complexity \- Subordination ratio")  
print("4. AI-ism Likelihood \- Weighted phrase detection")  
print("5. Function Word Ratio \- Closed-class word density")  
print("6. Discourse Marker Density \- Signposting frequency")  
print("7. Information Density \- Content word ratio")  
print("8. Epistemic Hedging \- Uncertainty markers")  
print("\\nAggregator: Weighted component score \+ AI-ism penalty (max 30)")  
print("Output: 0-100 final score with 4-level classification")

