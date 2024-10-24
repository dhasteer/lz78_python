from bitarray import bitarray
from bitarray.util import int2ba, ba2int
from lz78_python.utils.CharacterMap import CharacterMap
import numpy as np


def LZ78_decode(
        y: bitarray,
        length: int = None, # Desired length of the decoded sequence
        alphabet_size: int = 2,
        return_bitarray: bool = False,
        return_str: bool = False,
        custom_char_map: CharacterMap = None # for strings only
    ):
    """
    Decodes a sequence compressed via LZ78.
    """

    # Phrases that have been decoded so far
    phrases: list[list[int]] = []

    # Need to remove final leaf based on encoding assumption of complete phrase
    remove_leaf = bool(y[0])

    # Current index of the encoded bitarray
    i = 1

    # Number of phrases that have been decoded so far
    phrase_num = 0

    while i < len(y):
        # Matches the number of bits used to encode the corresponding phrase
        Lj = int(np.ceil(np.log2(phrase_num+1) + np.log2(alphabet_size)))
        comp_bits = ba2int(y[i:i+Lj])

        # Each decoded phrase consists of a prefix that is equal to one of the
        # previously-decoded phrases, plus one extra symbol. ref_idx indexes
        # the phrase corresponding to the prefix, and new_char is the final
        # symbol of the phrase.
        ref_idx = comp_bits // alphabet_size - 1
        new_char = comp_bits % alphabet_size
        if ref_idx >= 0:
            phrases.append(phrases[ref_idx] + [new_char])
        else:
            phrases.append([new_char])
        i += Lj
        phrase_num += 1

    symbols = []
    for phrase in phrases:
        symbols.extend(phrase)
    if remove_leaf:
       symbols = symbols[:-1]

    if return_str:
        if custom_char_map:
            return custom_char_map.decode(symbols)[:length]
        else:
            return "".join([chr(sym) for sym in symbols])

    if return_bitarray:
        x = bitarray()
        for sym in symbols:
            x.extend(int2ba(sym))
        return x[:length]
    return symbols[:length]
