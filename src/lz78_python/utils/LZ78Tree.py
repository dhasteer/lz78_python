from bitarray import bitarray
from typing import Union
from collections import namedtuple


class LZ78Tree:
    """
    Tree for LZ78 encoding, which is equivalent in structure to a Trie
    (https://en.wikipedia.org/wiki/Trie). Each node represents a phrase
    consisting of the path from the root to the node, where each branch is
    labeled with a character in the alphabet.
    
    The LZ78 tree is built during the parsing of a phrase as follows:
    1. Following the characters in the input string, traverse this tree until
        we reach a leaf.
    2. Add a new leaf corresponding to the next character in the input string.

    The phrase is defined as the path traversed in the tree, including the new leaf.
    
    By construction, each node of the tree corresponds to a phrase in the LZ78
    parsing of a string. We store the index of that phrase in the field
    `phrase_idx` (e.g., the first node added in the tree has `phrase_idx=0`, the
    second has `phrase_idx=1`, etc. So, the new leaf will have `phrase_idx` set
    to the number of phrases that have been parsed so far.)

    Each node has an additional variable, `seen_count`, which keeps track of
    the number of times the node has been reached in a traversal. This
    `seen_count` can, for instance, be used to create sequential probability
    assignments.
    """
    def __init__(self, alphabet_size=2, phrase_idx=-1):
        self.branches = {}
        self.alphabet_size = alphabet_size
        self.seen_count = 1
        self.phrase_idx = phrase_idx

    def __getitem__(self, index):
        return self.branches.get(int(index), None)
    
    def __setitem__(self, index, value):
        self.branches[int(index)] = value
    
def traverse_tree(x: Union[bitarray, list], root: LZ78Tree, start_idx: int, phrase_idx: int):
    """
    Traverse the LZ78 tree and add a new leaf, as described in the docstring of
    LZ78Tree.

    Inputs:
    - x: input symbols to parse
    - root: LZ78Tree to traverse
    - start_idx: index of the first character in `x` that we should parse.
    - phrase_idx: the number of phrases that have been encoded so far.
    """
    state:LZ78Tree = root
    end_idx = len(x)
    curr_idx = start_idx

    new_leaf = None
    for i in range(curr_idx, len(x)):
        # Traverse until we reach a leaf or the end of the tree
        if state[x[i]]:
            state = state[x[i]]
            state.seen_count += 1
        else:
            new_leaf = x[i]
            end_idx = i
            break

    if new_leaf is not None:
        # Add a new leaf to the LZ78 tree corresponding to the last symbol in the
        # phrase being parsed
        state[new_leaf] = LZ78Tree(phrase_idx=phrase_idx, alphabet_size=root.alphabet_size)
        phrase = x[start_idx:end_idx] + [new_leaf]
    else:
        phrase = x[start_idx:end_idx]
    
    # Return the full phrase, the node directly preceding the new leaf (or, if
    # we ran into the end of the input and no leaves were added, the last node
    # seen), and whether or not a new leaf was added
    LZ78TraverseOutput = namedtuple('LZ78TraverseOutput', ['phrase','state', 'full_phrase'])
    return LZ78TraverseOutput(phrase, state, new_leaf is not None)
