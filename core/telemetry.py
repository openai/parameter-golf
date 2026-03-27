import math

class Z0_Telemetry:
    """Calculates Wave Continuity Score (WCS) to reject noise dynamically."""
    def __init__(self, threshold=0.35):
        self.wcs_threshold = threshold
        self.known_attractors = set(range(4, 31)) # Indices of primal words in tokenizer

    def calculate_wcs(self, token_ids: list) -> float:
        if not token_ids: return 0.0
        
        # WCS is the ratio of coherent attractors to total signal volume
        coherent_signals = sum(1 for tid in token_ids if tid in self.known_attractors)
        wcs = coherent_signals / len(token_ids)
        
        return wcs

    def check_impedance(self, token_ids: list) -> bool:
        """Returns True if Z > 0 (high impedance / noise attack)."""
        return self.calculate_wcs(token_ids) < self.wcs_threshold
