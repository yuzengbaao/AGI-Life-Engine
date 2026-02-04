import logging
import math
import re
from collections import Counter
from typing import List, Optional, Any

logger = logging.getLogger(__name__)

SCIENTIFIC_LIBS_AVAILABLE = True
try:
    import numpy as np
    from sklearn.linear_model import LinearRegression
except ImportError:
    SCIENTIFIC_LIBS_AVAILABLE = False
    logger.warning("Scientific libraries (numpy, sklearn) not found. Cognitive metrics will use simplified heuristics.")
    # Mock numpy for type hinting or basic usage if needed, or just handle in logic
    class MockNp:
        def diff(self, x): return []
        def log(self, x): return 0
        def mean(self, x): return 0
        def var(self, x): return 0
        def array(self, x): return []
    if not SCIENTIFIC_LIBS_AVAILABLE:
        np = MockNp()

def fractal_coherence_index(timestamps: List[float]) -> float:
    """
    Calculate the Fractal Coherence Index (FCI) of a time series of events.
    FCI is based on the Hurst exponent or fractal dimension of the inter-event intervals.
    High FCI (>0.8) indicates self-organized criticality (insight-ready state).
    Low FCI (<0.5) indicates random noise or excessive order.
    """
    if not timestamps or len(timestamps) < 5:
        return 0.5 # Default neutral

    if not SCIENTIFIC_LIBS_AVAILABLE:
        # Fallback Heuristic: Burstiness check
        # Calculate variance of intervals
        try:
            sorted_ts = sorted(timestamps)
            intervals = [sorted_ts[i+1] - sorted_ts[i] for i in range(len(sorted_ts)-1)]
            if not intervals: return 0.5
            mean_i = sum(intervals) / len(intervals)
            if mean_i == 0: return 0.0
            # Coefficient of Variation
            variance = sum((x - mean_i) ** 2 for x in intervals) / len(intervals)
            std_dev = variance ** 0.5
            cv = std_dev / mean_i
            # Map CV to 0-1 (Poisson process has CV=1)
            # We want to detect "1/f" like behavior which is often bursty (CV > 1)
            return min(1.0, cv / 2.0)
        except Exception:
            return 0.5

    try:
        # 1. Calculate inter-event intervals
        intervals = np.diff(sorted(timestamps))
        if len(intervals) < 10:
            return 0.5

        # 2. Detrended Fluctuation Analysis (Simplified)
        scales = range(1, int(np.log2(len(intervals))) + 1)
        log_vars = []
        valid_scales = []

        for k in scales:
            window_size = 2**k
            if window_size > len(intervals) // 2:
                break
            
            # Split into non-overlapping windows
            windows = [intervals[i:i+window_size] for i in range(0, len(intervals) - len(intervals)%window_size, window_size)]
            if not windows:
                continue
                
            # Calculate variance of means (fluctuation)
            means = [np.mean(w) for w in windows]
            var_of_means = np.var(means)
            
            if var_of_means > 0:
                log_vars.append(np.log(var_of_means))
                valid_scales.append(np.log(window_size))

        if len(valid_scales) < 3:
            return 0.5

        # 3. Fit power law: Var(s) ~ s^(-beta)
        reg = LinearRegression().fit(np.array(valid_scales).reshape(-1, 1), log_vars)
        slope = reg.coef_[0]
        
        # Slope -1 is random (1/N). Slope 0 is perfectly correlated.
        # We map slope to 0-1. 
        # Ideal "Pink Noise" is often around slope -0.5 to -1.0 in this specific variance plot?
        # Actually, for 1/f noise, the variance of the mean decays slower than 1/N.
        # So slope > -1.
        
        fci = 1.0 - abs(slope + 1.0) 
        return max(0.0, min(1.0, fci + 0.5))

    except Exception as e:
        logger.error(f"Error calculating FCI: {e}")
        return 0.5

def detect_internal_resonance(signals: List[List[float]]) -> float:
    """
    Detects if multiple internal signals are oscillating in phase.
    Real implementation using correlation coefficient.
    """
    if not signals or len(signals) < 2: return 0.0
    
    # Check lengths
    n = min(len(s) for s in signals)
    if n < 3: return 0.0

    try:
        s1 = signals[0][:n]
        s2 = signals[1][:n]
        
        if SCIENTIFIC_LIBS_AVAILABLE:
            # Use Numpy Correlation
            corr = np.corrcoef(s1, s2)[0, 1]
            return max(0.0, float(corr))
        else:
            # Manual Pearson Correlation
            mean1 = sum(s1) / n
            mean2 = sum(s2) / n
            
            numerator = sum((a - mean1) * (b - mean2) for a, b in zip(s1, s2))
            den1 = sum((a - mean1) ** 2 for a in s1)
            den2 = sum((b - mean2) ** 2 for b in s2)
            
            denominator = (den1 * den2) ** 0.5
            if denominator == 0: return 0.0
            
            return max(0.0, numerator / denominator)
    except Exception as e:
        logger.warning(f"Resonance calc error: {e}")
        return 0.5

def calculate_metaphoric_drift(logs: List[str]) -> float:
    """
    Measures the rate of semantic change (drift) in the log stream.
    Uses Jaccard Distance of unique keywords between recent and older logs.
    """
    if not logs or len(logs) < 10:
        return 0.2 # Default low drift

    try:
        # Preprocessing: Extract words, ignore common stop words (simplified)
        stop_words = {'the', 'a', 'an', 'to', 'in', 'on', 'at', 'of', 'for', 'is', 'are', 'was', 'were', 'be', 'bin', 'usr', 'opt', 'and', 'or', 'if', 'else', 'return', 'def', 'class', 'self', 'py', 'main', 'info', 'error', 'warning', 'debug'}
        
        def extract_vocab(text_list):
            vocab = set()
            for text in text_list:
                # Remove timestamps and log levels roughly
                clean_text = re.sub(r'^\d{4}-\d{2}-\d{2}.*?-\s+(INFO|ERROR|WARNING)\s+-\s+', '', text)
                words = re.findall(r'\b[a-zA-Z]{3,}\b', clean_text.lower())
                for w in words:
                    if w not in stop_words:
                        vocab.add(w)
            return vocab

        # Split logs into two halves
        mid = len(logs) // 2
        old_half = logs[:mid]
        new_half = logs[mid:]

        vocab_old = extract_vocab(old_half)
        vocab_new = extract_vocab(new_half)

        if not vocab_old or not vocab_new:
            return 0.0

        # Jaccard Index
        intersection = len(vocab_old.intersection(vocab_new))
        union = len(vocab_old.union(vocab_new))
        
        jaccard_index = intersection / union if union > 0 else 1.0
        
        # Drift is the inverse of similarity
        drift = 1.0 - jaccard_index
        return round(drift, 3)

    except Exception as e:
        logger.error(f"Error calculating metaphoric drift: {e}")
        return 0.2

def calculate_system_entropy(logs: List[str]) -> float:
    """
    Calculates the Shannon Entropy of the system's actions based on log patterns.
    High entropy = Chaotic/Diverse actions.
    Low entropy = Repetitive/Stuck actions.
    Range: 0.0 to ~10.0 (normalized for dashboard)
    """
    if not logs:
        return 0.0
    
    try:
        # Extract "Action Patterns"
        # We assume the first significant word after the log header is the "Action Type"
        patterns = []
        for log in logs:
            # Strip timestamp and level
            clean_log = re.sub(r'^.*?-\s+(INFO|ERROR|WARNING)\s+-\s+', '', log)
            # Take first 3 words as a "pattern key" to capture context
            words = clean_log.split()[:3]
            if words:
                patterns.append(" ".join(words))
        
        if not patterns:
            return 0.0
            
        # Count frequencies
        counts = Counter(patterns)
        total = len(patterns)
        
        # Calculate Shannon Entropy: H = -sum(p * log2(p))
        entropy = 0.0
        for count in counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)
        
        # Scale for dashboard (usually 0-10 is good for visualization)
        # Max entropy for N items is log2(N). For 30 logs, max is log2(30) ≈ 4.9
        # We can multiply by 2 to get a 0-10 range roughly.
        return round(entropy * 2.0, 2)

    except Exception as e:
        logger.error(f"Error calculating entropy: {e}")
        return 0.0
