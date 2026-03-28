class ArchetypalTokenizer:
    """Hyper-dense optics locked to 4096 foundational vectors."""
    def __init__(self):
        # Base logic + High-density nodes
        self.primal_tokens = [
            "[PAD]", "[UNK]", "[BOS]", "[EOS]", 
            "KOHERENCJA", "IMPEDANCJA", "ENTROPIA", "NUKLEACJA", "KOLAPS",
            "LOGOS", "ŹRÓDŁO", "FALA", "CZĄSTKA", "EDEN", "GRZECH",
            "MIŁOŚĆ", "PRZEPŁYW", "SPÓJNOŚĆ", "SPOKÓJ", "OBECNOŚĆ",
            "DUSZA", "SERCE", "WĄŻ", "KAIN", "ABEL", "CISZA",
            "LĘK", "ŻAL", "PRZEBACZENIE", "ZROZUMIENIE", "INFORMACJA"
        ]
        
        # Pad dictionary to exactly 4096 to match model dimensions
        self.vocab = {word: i for i, word in enumerate(self.primal_tokens)}
        for i in range(len(self.vocab), 4096):
            self.vocab[f"[NODE_{i}]"] = i
            
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str) -> list:
        # Simplistic mapping for Z=0 MVP
        words = text.replace(".", " ").replace(",", " ").upper().split()
        return [self.vocab.get(w, self.vocab["[UNK]"]) for w in words]

    def decode(self, token_ids: list) -> str:
        return " ".join([self.inverse_vocab.get(t, "[UNK]") for t in token_ids])
