class CharacterMap:
    def __init__(self, x: str = None, A: int = 256):
        if x is not None:
            self.A = 0
            # Mapping alphabet char to representative integer symbol
            self.char_to_symbol = {}
            # Mapping (indirectly via index) integer symbol to alphabet char
            self.symbol_to_char = []
            for char in x:
                if char not in self.char_to_symbol:
                    self.char_to_symbol[char] = self.A
                    self.symbol_to_char.append(char)
                    self.A += 1
        else:
            # If x hasn't been specified, assumes `chr` and `ord` mappings
            # between integer symbols and alphabet characters.
            self.A = A
            self.symbol_to_char = None
            self.char_to_symbol = None

    def alphabet_size(self):
        return self.A
    
    def decode(self, symbols):
        # Return string corresponding inputted integer symbols based on mappings.
        if self.symbol_to_char:
            return "".join([self.symbol_to_char[sym] for sym in symbols])
        return "".join([chr(sym) for sym in symbols])
    
    def encode(self, x):
        # Return integer symbols corresponding inputted string based on mappings.
        if self.char_to_symbol:
            return [self.char_to_symbol[ch] for ch in x]
        return [ord(ch) for ch in x]

    def filter_string(self, x):
        # Takes inputted string and removes character that are not present in the character
        # mappings.
        x_filtered = ""
        for char in x:
            if char in self.char_to_symbol:
                x_filtered += char
        return x_filtered
        