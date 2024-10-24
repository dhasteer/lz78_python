from bitarray import bitarray
from bitarray.util import int2ba

from lz78_python.utils.LZ78Tree import LZ78Tree, traverse_tree
from lz78_python.utils.CharacterMap import CharacterMap
import numpy as np
from typing import Union

class BlockLZ78Encoder:
    """
    Class for performing LZ78 compression in a streamed fashion. After
    instantiating a BlockLZ78Encoder, you can call `encode_block`
    an arbitrary number of times such that the bitarray returned by
    `get_encoded_sequence` is equal to the output you would get from compressing the
    concatenation of all input blocks at once.
    """
    def __init__(self, alphabet_size: int = 2,
                 input_is_string: bool = False,
                 custom_char_map: CharacterMap = None):
        self.alphabet_size = alphabet_size

        # If the input is a string, we will use the CharacterMap class to
        # map integer symbols to ASCII characters. This lets us have custom
        # alphabets (e.g., lowercase letters and punctuation only), and easily
        # go back and forth between lists of integer symbols and strings
        self.input_is_string = input_is_string
        if self.input_is_string:
            if custom_char_map is not None:
                self.char_map = custom_char_map
                self.alphabet_size = custom_char_map.A
            else:
                # Default: just use built-in ord and chr functions
                self.char_map = CharacterMap(A=alphabet_size)

        self.tree = LZ78Tree(alphabet_size=self.alphabet_size)

        # If we're in the middle of parsing a phrase, where are we in the
        # LZ78 tree?
        self.state = self.tree

        # How many phrases have been parsed so far
        self.phrase_idx = 0
        
        self.total_uncompressed_bits = 0

        # If, after encoding a block, we're still in the middle of encoding
        # a phrase, we want to store that phrase and related information
        self.current_phrase = None
        self.current_phrase_ref_idx = None

        # Full output bitarray, except for any phrase currently being parsed
        self.output_stream_minus_current_phrase = bitarray()
        self.output_stream_minus_current_phrase.extend(int2ba(0))

        # Have we been given any content to encode
        self.empty = True

    def encode_block(self, x: Union[bitarray, list, str]):
        """
        Encode a block of symbols.
        """
        if isinstance(x, str):
            x = self.char_map.encode(x)
        elif self.input_is_string:
            raise RuntimeError("Expected string sequence with same character mapping")

        if not isinstance(x, list) and not isinstance(x, bitarray):
            x = list(x)

        # Used for computing compression ratio
        self.total_uncompressed_bits += len(x) * np.log2(self.alphabet_size)

        # The following variables are the same as in
        # LZ78_implementation.naive.encoder.
        start_idx = 0
        ref_idxs: list[int] = []
        output_leaves: list[int] = []

        # When constructing the output bitarray, we need to account for the
        # number of phrases we have parsed so far
        prev_phrase_idx = self.phrase_idx

        # Build the LZ78 tree
        while start_idx < len(x):
            phrase, state, full_phrase = traverse_tree(x, self.state, start_idx, self.phrase_idx)
            if self.current_phrase is not None:
                # Adjust for already being in the middle of encoding a phrase
                phrase = self.current_phrase + phrase
                x = self.current_phrase + x
                self.current_phrase = None

            if full_phrase:
                # We just parsed a full phrase
                self.phrase_idx += 1
                ref_idxs.append(state.phrase_idx)
                start_idx += len(phrase)
                output_leaves.append(x[start_idx-1])
                self.state = self.tree
            else:
                # We ran into the end of the input sequence and are still in
                # the middle of parsing a phrase.
                self.current_phrase = phrase
                self.current_phrase_ref_idx = state.phrase_idx
                self.state = state
                break
        
        if len(x) > 0:
            self.output_stream_minus_current_phrase[0] = int(not full_phrase)
            self.empty = False

        # encode the output string
        for i, leaf in enumerate(output_leaves):
            j = i + prev_phrase_idx # total number of phrases encoded so far
            Lj = int(np.ceil(np.log2(j+1) + np.log2(self.alphabet_size)))
            if j > 0:
                self.output_stream_minus_current_phrase.extend(
                    int2ba((ref_idxs[i]+1) * self.alphabet_size + int(leaf), Lj))
            else:
                self.output_stream_minus_current_phrase.extend(
                    int2ba(int(leaf), Lj))

    def get_encoded_sequence(self):
        """
        Returns a bitarray of representing the LZ78-compressed concatenation
        of all blocks processed so far.
        """
        if self.current_phrase is None:
            return self.output_stream_minus_current_phrase
        
        # If we are in the middle of parsing a phrase, we also need to encode
        # the phrase currently being parsed.
        Lj = int(np.ceil(np.log2(self.phrase_idx+1) + np.log2(self.alphabet_size)))
        return self.output_stream_minus_current_phrase + \
            int2ba((self.current_phrase_ref_idx+1) * self.alphabet_size, Lj)
    
    def compression_ratio(self):
        if self.empty:
            # Empty input to compress
            return 0

        if self.current_phrase is None:
            Lj = 0
        else:
            # Number of bits used to encode the phrase that is currently being
            # parsed
            Lj = int(np.ceil(np.log2(self.phrase_idx+1) + 
                             np.log2(self.alphabet_size)))

        return (len(self.output_stream_minus_current_phrase) + Lj) / \
                self.total_uncompressed_bits
        
    