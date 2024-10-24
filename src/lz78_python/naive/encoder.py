from bitarray import bitarray
from bitarray.util import int2ba

from lz78_python.utils.LZ78Tree import LZ78Tree, traverse_tree
from lz78_python.utils.CharacterMap import CharacterMap
from collections import namedtuple
import numpy as np
from typing import Union


def LZ78_encode(
        x: Union[bitarray, list, str],
        alphabet_size: int = 2,
        custom_char_map: CharacterMap = None # for strings only
) -> bitarray:
    """
    Encodes a sequence of symbols using LZ78, as follows:

    Repeat:
        1. Starting at the next unparsed symbol, traverse the LZ78 tree until
            we reach a leaf.
                By construction, that leaf was a previous phrase
                in the LZ78 parsing, and the LZ78Tree keeps track of the index
                of that phrase. We add that index to the output bitarray.
        2. Add a new leaf to the LZ78Tree corresponding to the next symbol
            in the input sequence. We add the value of that new symbol to the
            output bitarray.
        We call the full substring encoded in this iteration a phrase.

    It may be helpful to look at LZ78_implementation.utils.LZ78Tree,
    specifically the docstring of the LZ78Tree class.
    """
    if isinstance(x, str):
        if custom_char_map:
            x = custom_char_map.encode(x)
            alphabet_size = custom_char_map.A
        else:
            x = [ord(ch) for ch in x]
    if not isinstance(x, list) and not isinstance(x, bitarray):
        x = list(x)

    # LZ78 tree structure
    root = LZ78Tree(alphabet_size=alphabet_size)

    # ref_idxs stores the indices described in step 1 of the docstring above,
    # and output_leaves stors the new symbols described in step 2.
    ref_idxs: list[int] = []
    output_leaves: list[int] = []

    # Index of the input array
    start_idx = 0
    
    # How many phrases have been parsed so far
    phrase_idx = 0

    # Build the LZ78 tree
    while start_idx < len(x):
        phrase, state, full_phrase = traverse_tree(x, root, start_idx, phrase_idx)

        if full_phrase:
            phrase_idx += 1
            ref_idxs.append(state.phrase_idx)
            start_idx += len(phrase)
            output_leaves.append(x[start_idx-1])
        else:
            current_phrase_ref_idx = state.phrase_idx
            break
        
    # encode the output string
    output = bitarray()

    if len(x) > 0:
        output.extend(int2ba(int(not full_phrase)))

    for j, leaf in enumerate(output_leaves):
        # Encode each phrase, as described in page 4 of the LZ78 paper
        # (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1055934)
        Lj = int(np.ceil(np.log2(j+1) + np.log2(alphabet_size)))
        if j > 0:
            output.extend(int2ba((ref_idxs[j]+1) * alphabet_size + int(leaf), Lj))
        else:
            output.extend(int2ba(
                int(leaf), Lj
            ))

    if len(x) > 0 and not full_phrase:
        Lj = int(np.ceil(np.log2(phrase_idx+1) + np.log2(alphabet_size)))
        output += int2ba((current_phrase_ref_idx+1) * alphabet_size, Lj)
    
    compression_ratio = len(output) / (len(x) * np.log2(alphabet_size)) if len(x) > 0 else 0
    
    # Return the output bitarray and compression ratio
    CompressedSequence = namedtuple('CompressedSequence', ['bits', 'compression_ratio'])
    return CompressedSequence(output, compression_ratio)
    