import numpy as np
from scipy import special

import torch
from transformers import AutoTokenizer

from fernandez_llmwatermarking.hashing import get_seed_rng

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WmDetector():
    def __init__(self, 
        tokenizer: AutoTokenizer, 
        ngram: int = 1,
        seed: int = 0
    ):
        # model config
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        # watermark config
        self.ngram = ngram
        self.seed = seed
        self.rng = torch.Generator()
        self.rng.manual_seed(self.seed)

    def aggregate_scores(
        self, 
        scores: list[np.array], 
        aggregation: str = 'mean'
    ) -> float:
        """Aggregate scores along a text."""
        if aggregation == 'sum':
           return scores.sum(axis=0)
        elif aggregation == 'mean':
            return scores.mean(axis=0)
        elif aggregation == 'max':
            return scores.max(axis=0)
        else:
             raise ValueError(f'Aggregation {aggregation} not supported.')

    def get_details(
        self, 
        text: str,
        scoring_method: str="v2",
        ntoks_max: int = None,
    ) -> list[dict]:
        """
        Get score increment for each token in text.
        Args:
            text: input text
            scoring_method: 
                'none': score all ngrams
                'v1': only score tokens for which wm window is unique
                'v2': only score unique {wm window+tok} is unique
            ntoks_max: maximum number of tokens
        Output:
            token_details: list of dicts containing token info and scores
        """
        tokens_id = self.tokenizer.encode(text, add_special_tokens=False)
        if ntoks_max is not None:
            tokens_id = tokens_id[:ntoks_max]
        
        total_len = len(tokens_id)
        token_details = []
        seen_grams = set()
        
        # Add initial tokens that can't be scored (not enough context)
        num_start = min(self.ngram, total_len)
        for i in range(num_start):
            token_details.append({
                'token_id': tokens_id[i],
                'is_scored': False,
                'score': float('nan'),
                'token_text': self.tokenizer.decode([tokens_id[i]])
            })
        
        # Score remaining tokens
        for cur_pos in range(self.ngram, total_len):
            ngram_tokens = tokens_id[cur_pos-self.ngram:cur_pos]
            is_scored = True
            
            if scoring_method == 'v1':
                tup_for_unique = tuple(ngram_tokens)
                is_scored = tup_for_unique not in seen_grams
                if is_scored:
                    seen_grams.add(tup_for_unique)
            elif scoring_method == 'v2':
                tup_for_unique = tuple(ngram_tokens + [tokens_id[cur_pos]])
                is_scored = tup_for_unique not in seen_grams
                if is_scored:
                    seen_grams.add(tup_for_unique)
                    
            score = float('nan')
            if is_scored:
                score = self.score_tok(ngram_tokens, tokens_id[cur_pos])
                score = float(score)
                
            token_details.append({
                'token_id': tokens_id[cur_pos],
                'is_scored': is_scored,
                'score': score,
                'token_text': self.tokenizer.decode([tokens_id[cur_pos]])
            })
            
        return token_details

    def get_pvalues_by_tok(
        self, 
        token_details: list[dict]
    ) -> tuple[list[float], dict]:
        """
        Get p-value for each token so far.
        Args:
            token_details: list of dicts containing token info and scores from get_details()
        Returns:
            tuple containing:
            - list of p-values, with nan for unscored tokens
            - dict with auxiliary information:
                - final_score: final running score
                - ntoks_scored: final number of scored tokens
                - final_pvalue: last non-nan pvalue (0.5 if none available)
        """
        pvalues = []
        running_score = 0
        ntoks_scored = 0
        eps = 1e-10  # small constant to avoid numerical issues
        last_valid_pvalue = 0.5  # default value if no tokens are scored
        
        for token in token_details:
            if token['is_scored']:
                running_score += token['score']
                ntoks_scored += 1
                pvalue = self.get_pvalue(running_score, ntoks_scored, eps)
                last_valid_pvalue = pvalue
                pvalues.append(pvalue)
            else:
                pvalues.append(float('nan'))
        
        aux_info = {
            'final_score': running_score,
            'ntoks_scored': ntoks_scored,
            'final_pvalue': last_valid_pvalue
        }
            
        return pvalues, aux_info

    def score_tok(self, ngram_tokens: list[int], token_id: int):
        """ for each token in the text, compute the score increment """
        raise NotImplementedError
    
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ compute the p-value for a couple of score and number of tokens """
        raise NotImplementedError


class MarylandDetector(WmDetector):

    def __init__(self, 
            tokenizer: AutoTokenizer,
            ngram: int = 1,
            seed: int = 0,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = 1 if token_id in greenlist else 0
        """
        seed = get_seed_rng(self.seed, ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n toks in the greenlist
        scores[greenlist] = 1 
        return scores[token_id]
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a binomial distribution """
        pvalue = special.betainc(score, 1 + ntoks - score, self.gamma)
        return max(pvalue, eps)

class MarylandDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: AutoTokenizer,
            ngram: int = 1,
            seed: int = 0,
            gamma: float = 0.5, 
            delta: float = 1.0, 
            **kwargs):
        super().__init__(tokenizer, ngram, seed, **kwargs)
        self.gamma = gamma
        self.delta = delta
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as MarylandDetector but using zscore """
        seed = get_seed_rng(self.seed, ngram_tokens)
        self.rng.manual_seed(seed)
        scores = torch.zeros(self.vocab_size)
        vocab_permutation = torch.randperm(self.vocab_size, generator=self.rng)
        greenlist = vocab_permutation[:int(self.gamma * self.vocab_size)] # gamma * n
        scores[greenlist] = 1
        return scores[token_id]
                
    def get_pvalue(self, score: int, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        zscore = (score - self.gamma * ntoks) / np.sqrt(self.gamma * (1 - self.gamma) * ntoks)
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)
    
class OpenaiDetector(WmDetector):

    def __init__(self, 
            tokenizer: AutoTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ 
        score_t = -log(1 - rt[token_id]])
        """
        seed = get_seed_rng(self.seed, ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log()
        return scores[token_id]
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a gamma distribution """
        pvalue = special.gammaincc(ntoks, score)
        return max(pvalue, eps)

class OpenaiDetectorZ(WmDetector):

    def __init__(self, 
            tokenizer: AutoTokenizer, 
            ngram: int = 1,
            seed: int = 0,
            **kwargs):
        super().__init__(tokenizer, ngram, seed, **kwargs)
    
    def score_tok(self, ngram_tokens, token_id):
        """ same as OpenaiDetector but using zscore """
        seed = get_seed_rng(self.seed, ngram_tokens)
        self.rng.manual_seed(seed)
        rs = torch.rand(self.vocab_size, generator=self.rng) # n
        scores = -(1 - rs).log()
        return scores[token_id]
 
    def get_pvalue(self, score: float, ntoks: int, eps: float):
        """ from cdf of a normal distribution """
        mu0 = 1
        sigma0 = np.pi / np.sqrt(6)
        zscore = (score/ntoks - mu0) / (sigma0 / np.sqrt(ntoks))
        pvalue = 0.5 * special.erfc(zscore / np.sqrt(2))
        return max(pvalue, eps)