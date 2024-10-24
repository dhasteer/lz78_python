from bitarray import bitarray
from typing import Union
from collections import namedtuple
from lz78_python.utils.LZ78Tree import LZ78Tree
import numpy as np


def compute_spa(
        state_count: int, 
        next_state_count: int,
        alphabet_size: int,
        gamma: float, 
):
    """
    Computes sequential probability assignment of the next symbol given the 
    current state, i.e., the current node of the LZ78 tree, according to the
    Dirichlet SPA described in Construction III.6 of the SW24 paper:
    https://arxiv.org/pdf/2410.06589. 

    Inputs:
    - state_count: the number of times that we have previously traversed the
        current node of the LZ78 tree
    - next_state_count: the number of times that we have previously traversed
        the node corresponding to the next symbol
    - alphabet_size: number of possible symbols
    - gamma: value between 0 and 1 used as the Dirichlet parameter for the SPA
    """
    return (next_state_count + gamma) / \
            (state_count - 1 + gamma * alphabet_size)


def traverse_tree_and_compute_loss(
    x: Union[bitarray, list],
    root: LZ78Tree,
    start_idx: int,
    phrase_idx: int,
    gamma: float, 
    keep_log_loss_history: bool = False,
    tree_frozen: bool = False,
    tree_size_frozen: bool = False,
):
    """
    Traverse the LZ78 tree and add a new leaf, as described in the docstring of
    LZ78Tree.

    Inputs:
    - x: input symbols to parse
    - root: LZ78Tree to traverse
    - start_idx: index of the first character in `x` that we should parse.
    - phrase_idx: the number of phrases that have been encoded so far.
    - gamma: value between 0 and 1 used as the Dirichlet parameter for the SPA
        calculated as Construction III.6 in https://arxiv.org/pdf/2410.06589
    - keep_log_loss_history: whether to keep track of and return the per-symbol
        log loss as a list
    - tree_frozen: if true, the "seen_count" of each node will not be updated,
        nor will new leaves be added to the LZ78 tree.
    - tree_size_frozen: if true, the "seen_count" of each node will still be
        updated (unless tree_frozen is also set), but no new leaves will be
        added. This is useful in limiting memory usage.
    """
    state:LZ78Tree = root
    end_idx = len(x)
    curr_idx = start_idx

    log_losses = []
    total_log_loss = 0

    # If the tree is frozen, its size is frozen by default
    tree_size_frozen = tree_size_frozen or tree_frozen

    new_leaf = None
    for i in range(curr_idx, len(x)):
        # Traverse until we reach a leaf or the end of the tree
        if state[x[i]]:
            next_state_count = state[x[i]].seen_count
            next_state = state[x[i]]
        else:
            new_leaf = x[i]
            end_idx = i
            next_state_count = 0

        # Compute the sequential probability assignment
        spa_value = compute_spa(
            state_count=state.seen_count,
            next_state_count=next_state_count,
            alphabet_size=state.alphabet_size,
            gamma=gamma,
        )

        # Update the seen count *after* computing SPA!
        if not tree_frozen:
            state.seen_count += 1
        
        instantaneous_log_loss = np.log2(1/spa_value)
        if keep_log_loss_history:
            log_losses.append(instantaneous_log_loss)
        total_log_loss += instantaneous_log_loss

        if new_leaf is not None:
            break
        state = next_state

    if new_leaf is not None and not tree_size_frozen:
        # Add a new leaf to the LZ78 tree corresponding to the last symbol in the
        # phrase being parsed
        state[new_leaf] = LZ78Tree(phrase_idx=phrase_idx, alphabet_size=root.alphabet_size)
        phrase = x[start_idx:end_idx] + [new_leaf]
    else:
        phrase = x[start_idx:end_idx]
    
    # Return:
    # - phrase: all symbols parsed by this function call
    # - state: the node directly preceding the new leaf (or, if we ran into the
    #        end of the input and no leaves were added, the last node seen)
    # - full_phrase: a boolean for whether we reached the leaf of the tree
    # - total_log_loss: total log loss incurred by the SPA for this phrase
    # - log_losses: list of per-symbol log losses
    SPATraversalOutput = namedtuple('SPATraversalOutput', [
        'phrase','state', 'full_phrase', 'total_log_loss', 'log_losses'])
    return SPATraversalOutput(
        phrase, state, new_leaf is not None,
        total_log_loss, log_losses)
    