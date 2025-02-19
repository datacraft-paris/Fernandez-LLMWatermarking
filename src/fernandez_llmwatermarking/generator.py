
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from fernandez_llmwatermarking.hashing import get_seed_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmGenerator():
    def __init__(self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        ngram: int = 1,
        seed: int = 0,
        **kwargs
    ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.model = model
        self.max_seq_len = model.config.max_sequence_length if 'max_sequence_length' in model.config.to_dict() else 2048
        self.pad_id = model.config.pad_token_id if model.config.pad_token_id is not None else -1
        self.eos_id = model.config.eos_token_id
        # watermark config
        self.ngram = ngram
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_gen_len: int,
        temperature: float = 0.8,
        top_p: float = 0.95,
        return_aux: bool = False,
    ) -> str:
        
        prompt_tokens = self.tokenizer.encode(prompt)
        prompt_size = len(prompt_tokens)
        total_len = min(self.max_seq_len, max_gen_len + prompt_size)
        tokens = torch.full((1, total_len), self.pad_id).to(device).long()
        if total_len < prompt_size:
            print("prompt is bigger than max sequence length")
            prompt_tokens = prompt_tokens[:total_len]
        tokens[0, :len(prompt_tokens)] = torch.tensor(prompt_tokens).long()
        input_text_mask = tokens != self.pad_id

        start_pos = prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            past_key_values = outputs.past_key_values if prev_pos > 0 else None
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], 
                use_cache=True, 
                past_key_values=past_key_values
            )
            ngram_tokens = tokens[0, cur_pos-self.ngram:cur_pos].tolist()
            aux = {
                'ngram_tokens': ngram_tokens,
                'cur_pos': cur_pos,
            }
            next_tok = self.sample_next(outputs.logits[:, -1, :], aux, temperature, top_p)
            tokens[0, cur_pos] = torch.where(input_text_mask[0, cur_pos], tokens[0, cur_pos], next_tok)
            prev_pos = cur_pos
            if next_tok == self.eos_id:
                break

        # cut to max gen len
        t = tokens[0, :prompt_size + max_gen_len].tolist()
        # cut to eos tok if any
        finish_reason = 'length'
        try:
            find_eos = t[prompt_size:].index(self.eos_id)
            if find_eos:
                t = t[: prompt_size+find_eos]
            finish_reason = 'eos'
        except ValueError:
            pass
        aux_info = {
            't': t, 
            'finish_reason': finish_reason,
            'n_toks_gen': len(t) - prompt_size,
            'n_toks_tot': len(t),
        }
        decoded = self.tokenizer.decode(t)

        if return_aux:
            return decoded, aux_info
        return decoded
    
    def sample_next(
        self,
        logits: torch.FloatTensor, # (1, vocab_size): logits for last token
        aux: dict, # ngram_tokens (1, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        """Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)[0]  # Get the single token value
        return next_token


class OpenaiGenerator(WmGenerator):
    """
    Generate text using LLaMA and Aaronson's watermarking method.
    From ngram tokens, select the next token based on the following:
    - hash the ngram tokens and get a seed
    - use the seed to generate V random number r between [0,1]
    - select argmax ( r^(1/p) )
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def sample_next(
        self,
        logits: torch.FloatTensor, # (1, vocab_size): logits for last token
        aux: dict, # (1, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        """
        Generates the next token based on the provided logits by applying top-p (nucleus) sampling
        with an adjustable temperature. The process is detailed line by line below.
        1. Extracts the n-gram tokens sequence from the auxiliary dictionary `aux`.
        2. If the temperature is greater than 0, a probabilistic (random) sampling is applied.
        - Otherwise, the token with the highest logit is selected deterministically.
        3. Applies the softmax function to the logits scaled by the temperature to obtain a probability distribution.
        4. Sorts the probabilities in descending order.
        - `probs_sort` contains the sorted probabilities.
        - `probs_idx` contains the indices corresponding to the tokens in sorted order.
        5. Computes the cumulative sum along the last dimension of the sorted probabilities.
        6. Generates a boolean mask identifying tokens where the cumulative sum (excluding the token's own probability) exceeds the top-p threshold.
        7. Sets the probabilities that do not satisfy the top-p criterion to zero.
        8. Divides the filtered probabilities by their sum to ensure a normalized distribution.
        9. Computes a seed value based on the initial seed and the n-gram tokens for reproducibility.
        10. Sets the seed for the random number generator `self.rng`.
        11. Generates a tensor of random numbers between 0 and 1 for each token in the vocabulary.
        12. Ensures that the tensor `rs` is on the same device (CPU/GPU) as `probs_sort`.
        13. Reorders the random values to correspond with the ordering of tokens in `probs_sort`.
        14. Raises the random numbers to the power of the inverse of the sorted probabilities.
         - This transformation prepares the token selection process by emphasizing the relative differences.
        15. Identifies the index of the token with the highest value in the transformed `probs_sort`.
        16. Retrieves the original index of the selected token from the sorted indices.
        17. If the temperature is 0, directly selects the token with the highest logit without random sampling.
        18. Reshapes the tensor to extract a single value representing the next token.
        19. Returns the next token determined by either top-p sampling or deterministic selection.
        Parameters
        ----------
        logits : torch.FloatTensor
            A tensor of shape (1, vocab_size) containing the logits for the last token in the sequence.
        aux : dict
            A dictionary containing auxiliary data needed for sampling. It must include:
                - 'ngram_tokens': Tokens used to seed the random number generator for deterministic sampling.
        temperature : float, optional (default=0.8)
            Controls the randomness of the sampling:
                - A higher temperature (>0) yields a more random (softer) probability distribution.
                - If set to 0 or below, the function uses greedy sampling (argmax).
        top_p : float, optional (default=0.95)
            The cumulative probability threshold for nucleus (top-p) sampling. Only the top tokens with a
            cumulative probability less than or equal to `top_p` are considered for sampling.
        Returns
        -------
        int
            The index of the sampled next token.
        """
        ngram_tokens = aux['ngram_tokens']
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            # seed with hash of ngram tokens
            seed = get_seed_rng(self.seed, ngram_tokens)
            self.rng.manual_seed(seed)
            # generate rs randomly between [0,1]
            rs = torch.rand(self.vocab_size, generator=self.rng) # n
            rs = torch.Tensor(rs).to(probs_sort.device)
            rs = rs[probs_idx[0]] 
            # compute r^(1/p)
            probs_sort[0] = torch.pow(rs, 1/probs_sort[0])
            # select argmax ( r^(1/p) )
            next_token = torch.argmax(probs_sort, dim=-1, keepdim=True)
            next_token = torch.gather(probs_idx, -1, next_token)
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)[0]  # Get the single token value
        return next_token


class MarylandGenerator(WmGenerator):
    """
    Generate text using LLaMA and Maryland's watemrarking method.
    From ngram tokens, select the next token based on the following:
    - hash the ngram tokens and get a seed
    - use the seed to partition the vocabulary into greenlist (gamma*V words) and blacklist 
    - add delta to greenlist words' logits
    """
    def __init__(self, 
            *args, 
            gamma: float = 0.5,
            delta: float = 1.0,
            **kwargs
        ):
        super().__init__(*args, **kwargs)        
        self.gamma = gamma
        self.delta = delta

    def sample_next(
        self,
        logits: torch.FloatTensor, # (1, vocab_size): logits for last token
        aux: dict, # ngram_tokens (1, ngram): tokens to consider when seeding
        temperature: float = 0.8, # temperature for sampling
        top_p: float = 0.95, # top p for sampling
    ):
        """
        1. Extracts the n-gram tokens from the auxiliary dictionary `aux`.
        2. Applies a custom logits processing function (possibly to modify or mask logits) based on the n-gram tokens.
        3. If temperature is greater than 0, use probabilistic sampling.
        4. Applies the softmax function to the scaled logits (scaled by the temperature) to obtain a probability distribution
        over the vocabulary.
        5. Sorts the probabilities from highest to lowest.
        - `probs_sort` contains the sorted probabilities.
        - `probs_idx` contains the indices corresponding to the tokens in the sorted order.
        6. Calculates the cumulative sum of the sorted probabilities along the last dimension.
        7. Generates a boolean mask that identifies tokens where the cumulative sum, excluding the token's own probability,
        exceeds the top-p threshold.
        8. Sets the probabilities for tokens outside the top-p threshold to zero.
        9. Divides the filtered probabilities by their sum to ensure that the remaining probabilities form a valid distribution.
        10. Uses multinomial sampling to draw one sample from the renormalized distribution. The result corresponds to an index
        in the sorted probability tensor.
        11. Uses the sorted indices (`probs_idx`) to retrieve the original vocabulary index corresponding to the sampled token.
        12. If the temperature is 0, no sampling is performed; the token with the highest logit is selected directly.
        13. Reshapes the tensor to obtain a single scalar token value.
        14. Returns the next token, either sampled probabilistically (if temperature > 0) or chosen deterministically.
        Parameters
        ----------
        logits : torch.FloatTensor
            A tensor of shape (1, vocab_size) containing the unnormalized log probabilities (logits) for the
            last generated token.
        aux : dict
            A dictionary of auxiliary inputs. It must contain the key 'ngram_tokens' whose value is a tensor of
            tokens (shape: (1, ngram)) that are used to seed and potentially modify the token selection process.
        temperature : float, optional (default=0.8)
            A positive float value used to scale the logits. A higher temperature increases randomness, whereas a
            temperature of 0 (or a non-positive value) makes the selection deterministic by choosing the token with the
            highest logit.
        top_p : float, optional (default=0.95)
            A float between 0 and 1 used for nucleus (top-p) sampling. After applying softmax to the scaled logits,
            tokens are sorted by probability, and only the smallest set of tokens whose cumulative probability exceeds
            top_p are considered for sampling. The probabilities of the other tokens are set to zero.
        Returns
        -------
        torch.Tensor
            A single token (scalar tensor) representing the next token sampled according to the adjusted
            probability distribution.
        """
        ngram_tokens = aux['ngram_tokens']
        logits = self.logits_processor(logits, ngram_tokens)
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1) # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token) # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)[0]  # Get the single token value
        return next_token

    def logits_processor(self, logits, ngram_tokens):
        """Process logits to mask out words in greenlist."""
        logits = logits.clone()
        seed = get_seed_rng(self.seed, ngram_tokens)
        self.rng.manual_seed(seed)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        bias = torch.zeros(self.vocab_size).to(logits.device)
        bias[greenlist] = self.delta
        logits[0] += bias # add bias to greenlist words
        return logits
