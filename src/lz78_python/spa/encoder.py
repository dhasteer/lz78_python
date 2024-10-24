from bitarray import bitarray
from bitarray.util import int2ba

from lz78_python.streamed.encoder import BlockLZ78Encoder
from lz78_python.utils.CharacterMap import CharacterMap
from lz78_python.spa.ComputePhraseLoss import traverse_tree_and_compute_loss, compute_spa
import numpy as np
from typing import Union
import numpy as np


class LZ78SPA(BlockLZ78Encoder):
    """
    Class for computing sequential probability assignments based on the LZ78
    tree. This class extends BlockLZ78Encoder, so it also acts as an LZ78
    compressor.


    """
    def __init__(
        self,
        gamma: float = 0.5, 
        alphabet_size: int = 2,
        keep_log_loss_history: bool = False,
        input_is_string: bool = False,
        custom_char_map: CharacterMap = None
    ):
        """
        Inputs:
        - alphabet_size: number of distinct characters in the input
        - gamma: value between 0 and 1 used as the Dirichlet parameter  
            for the SPA calculated as Construction III.6 in 
            https://arxiv.org/pdf/2410.06589
        - keep_log_loss_history: whether to keep track of the per-symbol log
            loss as a list..
        - input_is_string: whether the input sequences will be strings of text.
        - custom_char_map: if the input sequences are strings, you can specify
            a custom character map, which specifies how characters are mapped to
            integer symbols. See LZ78_implementation.utils.CharacterMap for
            implementation details.
        """
        super().__init__(alphabet_size, input_is_string=input_is_string,
                         custom_char_map=custom_char_map)
        self.gamma = gamma

        # number of symbols processed
        self.n = 0
        self.total_log_loss = 0

        self.keep_log_loss_history = keep_log_loss_history
        self.log_loss_history = []

    def sample_index_from_dist(self, probabilities):
        """
        Samples from the discrete probability distribution specified by the
        input, and returns the index of the sample.
        """
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1 # in case of FP error
        rand = np.random.random()
        return int(np.where(rand < cdf)[0][0])

    def generate_data(
            self,
            n: int,
            min_context: int=1,
            temperature: float=0.1,
            top_k: int=0,
            seed_data: Union[list[int],bitarray,str]=None
        ):
        """
        Generates a sequence of data, using temperature and top-k sampling (see
        the "Experiments" section of [Sagan and Weissman 2024] for more details).

        Inputs:
        - n: number of symbols to generate
        - min_context: the SPA tries to maintain a context of at least a
            certain length at all times. So, when we reach a leaf of the LZ78
            prefix tree, we try traversing the tree with different suffixes of
            the generated sequence until we get a sufficiently long context
            for the next symbol.
        - temperature: a measure of how "random" the generated sequence is. A
            temperature of 0 deterministically generates the most likely
            symbols, and a temperature of 1 samples directly from the SPA.
            Temperature values around 0.1 or 0.2 function well.
        - top_k: forces the generated symbols to be of the top_k most likely
            symbols at each timestep.
        - seed_data: you can specify that the sequence of generated data
            be the continuation of the specified sequence.

        Returns a tuple of the generated sequence and that sequence's log loss,
        or perplexity.

        Errors if the SPA has not been trained so far, or if the seed data is
        not over the same alphabet as the training data.
        """
        x = [] # output symbols
        total_log_loss = 0

        old_state = self.state

        if seed_data:
            # This traverses the LZ78 tree with the seed_data, without
            # updating the tree at all
            self.state=self.tree
            _, self.state = self.compute_test_loss(
                seed_data, length=len(seed_data), start_idx=0, return_state=True
            )

        for sample_num in range(n):
            if self.state == self.tree or len(self.state.branches) == 0:
                # We've reached a leaf and need to re-seed the SPA with a
                # context, i.e., we start at the root and traverse the LZ78
                # tree with a snippet of the most recent output tokens
                for i in range(min(min_context, sample_num), -1, -1):
                    if i == 0:
                        self.state = self.tree
                    else:
                        traverse_output = traverse_tree_and_compute_loss(
                            x=x, 
                            root=self.tree, 
                            start_idx=sample_num-i,
                            phrase_idx=self.phrase_idx, 
                            gamma=self.gamma,
                            tree_frozen=True
                        )
                        _, self.state, reset_state, _, _ = traverse_output

                        if reset_state:
                            self.state = self.tree
                        
                    # successful re-seeding
                    if self.state != self.tree and len(self.state.branches) != 0:
                        break
        
            # Compute the SPA and perturb it using the temperature and top_k
            spa = np.array([self.get_prob_for_next_symbol(i) for i in range(self.alphabet_size)])

            if temperature == 0:
                # deterministic: pick the most-probable symbol
                # picks greatest possible argmax index
                x.append(np.where(spa == spa.max())[0][-1])            
            else:
                if temperature == 1:
                    probabilities = spa
                else:
                    logits = [np.log2(p) for p in spa]
                    # Using the following definition of temperature:
                    # P(token_i) = exp(logits(token_i) / T) / Σ_j exp(logits(token_j) / T) 
                    probabilities = np.array([2**(logit/temperature) for logit in logits])
            
                top_k = min(self.alphabet_size, top_k)
                if top_k > 0:
                    top_k_indices = np.argpartition(probabilities, -top_k)[-top_k:]
                    mask = np.zeros_like(probabilities, dtype=bool)
                    mask[top_k_indices] = True
                    top_k_probabilities = np.where(mask, probabilities, 0)
                    top_k_probabilities /= sum(top_k_probabilities)
                else:
                    top_k_probabilities = probabilities
            
                # randomly sample the next symbol
                x.append(self.sample_index_from_dist(top_k_probabilities))

            phrase_log_loss, self.state = self.compute_test_loss(
                x, length=1, start_idx = sample_num, return_state=True, 
            )
            total_log_loss += phrase_log_loss

        # Reset self.state
        self.state = old_state
        if self.input_is_string:
            x = self.char_map.decode(x)
        return x, total_log_loss

    def get_prob_for_next_symbol(self, sym: Union[int, str]):
        """
        Evaluate the sequential probability assignment, for the context given by
        `self.state`
        """
        if isinstance(sym, str):
            assert len(sym) == 1
            sym = self.char_map.encode(sym)
        next_state_count = 0 if not self.state[sym] else self.state[sym].seen_count
        return compute_spa(
            self.state.seen_count, 
            next_state_count,
            alphabet_size=self.alphabet_size, 
            gamma=self.gamma,
        )

    def compute_test_loss(
        self, x: Union[bitarray, list, str],
        length,
        start_idx=0,
        return_state=False,
        include_prev_context=True,
    ):
        """
        Traverses the LZ78 tree according to the input sequence and computes
        log loss, without building the tree or updating the counts of the 
        number of times we traverse each node. This also doesn't update
        `self.state` or any other object attributes. 
        """
        if isinstance(x, str):
            x = self.char_map.encode(x)
        if not isinstance(x, list) and not isinstance(x, bitarray):
            x = list(x)
        
        t = self.n + 1
        total_log_loss = 0
        end_idx = start_idx + length
        state = (self.state if include_prev_context else self.tree)
        while start_idx < end_idx:
            traverse_output = traverse_tree_and_compute_loss(
                x=x, root=state, start_idx=start_idx,
                phrase_idx=self.phrase_idx, 
                gamma=self.gamma,
                tree_frozen=True
            )
            phrase, state, reset_state, phrase_log_loss, _ = traverse_output
            if reset_state:
                state = self.tree
            total_log_loss += phrase_log_loss
            start_idx += (len(phrase) + 1)
        if return_state:
            return total_log_loss, state
        return total_log_loss

    def train_on_block(self, x: Union[bitarray, list, str], include_prev_context: bool = False):
        """
        Encode a block of symbols, and compute the log loss.
        """
        if self.input_is_string:
            x = self.char_map.encode(x)
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
        t = self.n + 1
        
        state = (self.state if include_prev_context else self.tree)
        prev_log_loss = self.total_log_loss

        # Build the LZ78 tree
        while start_idx < len(x):
            traverse_output = traverse_tree_and_compute_loss(
                x=x, root=state, 
                start_idx=start_idx,
                phrase_idx=self.phrase_idx,
                gamma=self.gamma,
                keep_log_loss_history=self.keep_log_loss_history
            )
            phrase, state, full_phrase, phrase_log_loss, log_losses = traverse_output
            t += len(phrase)
            self.total_log_loss += phrase_log_loss
            if self.keep_log_loss_history:
                self.log_loss_history.extend(log_losses)

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
                self.state, state = self.tree, self.tree
            else:
                # We ran into the end of the input sequence and are still in
                # the middle of parsing a phrase.
                self.current_phrase = phrase
                self.current_phrase_ref_idx = state.phrase_idx
                self.state = state
                break
        self.n = t - 1

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
        
        return self.total_log_loss - prev_log_loss
                
    def get_normalized_log_loss(self):
        return self.total_log_loss / self.n
