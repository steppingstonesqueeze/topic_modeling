#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HDP-LDA (direct-assignment, collapsed Gibbs) — tiny, self-contained demo
References:
- Teh, Jordan, Beal, Blei (2006) "Hierarchical Dirichlet Processes" (JASA)
- Antoniak distribution for table counts

Run:
    python hdp_lda_demo.py

No external dependencies.
"""

import math, random, re
from collections import Counter

# ------------- Data -------------
DOCS_RAW = [
    # 3 user-provided NYC/sunny texts
    "it is sunny in NYC today",
    "NYC has amazing restaurants in many parks when its sunny",
    "I wish I knew Urdu",
    # 5 puffer-fish texts
    "Puffer fish contain tetrodotoxin and require careful preparation by licensed chefs",
    "Marine biologists study the spines toxins and inflation behavior of pufferfish",
    "In Japanese cuisine fugu is a delicacy made from puffer fish served as sashimi",
    "The puffer fish inflates to deter predators and has bright warning colors",
    "Toxins in pufferfish can cause paralysis and respiratory failure if mishandled",
]

def tokenize(s):
    toks = re.findall(r"[A-Za-z]+", s.lower())
    stop = {"is","in","has","when","its","i","the","and","as","from","of","a","to","by","an","can","if"}
    return [t for t in toks if t not in stop]

# ------------- Hyperparameters -------------
ETA   = 0.5   # topic-word Dirichlet
ALPHA = 1.0   # doc-level concentration
GAMMA = 1.0   # global concentration

SEED = 7
ITERS = 200
TOPN  = 8

# ------------- Sampler Implementation -------------
class HDPLDADirectAssign:
    def __init__(self, docs_tokens, eta=0.5, alpha=1.0, gamma=1.0, seed=7):
        random.seed(seed)
        self.docs    = docs_tokens
        self.D       = len(docs_tokens)
        self.vocab   = sorted({w for d in docs_tokens for w in d})
        self.V       = len(self.vocab)
        self.w2i     = {w:i for i,w in enumerate(self.vocab)}
        self.corpus  = [[self.w2i[w] for w in d] for d in docs_tokens]

        self.eta   = eta
        self.alpha = alpha
        self.gamma = gamma

        # State
        self.K     = 0
        self.n_dk  = [Counter() for _ in range(self.D)]  # per-doc topic counts
        self.n_kw  = []   # per-topic word counts
        self.n_k   = []   # per-topic token totals
        self.beta  = []   # topic weights
        self.beta_rest = 1.0
        self.z     = [[-1]*len(doc) for doc in self.corpus]  # token topic assignments

    @staticmethod
    def _antoniak_draw(n, a):
        """m ~ Antoniak(n, a) via Bernoulli-sum trick."""
        if n <= 0: return 0
        m = 0
        for i in range(1, n+1):
            p = a / (a + i - 1.0)
            if random.random() < p: m += 1
        return m

    @staticmethod
    def _dirichlet_sample(params):
        draws = [random.gammavariate(max(x,0.0)+1e-9, 1.0) for x in params]
        S = sum(draws)
        if S == 0:  # fallback uniform-ish
            k = len(params)
            return [1.0/k]*k
        return [d/S for d in draws]

    def _ensure_topic(self):
        # stick break: u ~ Beta(1, gamma)
        u = random.betavariate(1.0, self.gamma)
        new_weight = self.beta_rest * u
        self.beta_rest *= (1.0 - u)
        self.beta.append(new_weight)
        self.n_kw.append(Counter()); self.n_k.append(0)
        self.K += 1
        return self.K-1

    def _maybe_compact(self):
        keep = [k for k in range(self.K) if self.n_k[k] > 0]
        if len(keep) == self.K: return
        mapping = {old:new for new,old in enumerate(keep)}
        removed_mass = sum(self.beta[k] for k in range(self.K) if k not in keep)
        self.beta_rest += removed_mass
        self.beta = [self.beta[k] for k in keep]
        self.n_kw = [self.n_kw[k] for k in keep]
        self.n_k  = [self.n_k[k]  for k in keep]
        for d in range(self.D):
            self.n_dk[d] = Counter({mapping[k]:v for k,v in self.n_dk[d].items() if k in mapping})
            for n, zk in enumerate(self.z[d]):
                if zk != -1:
                    self.z[d][n] = mapping.get(zk, -1)
        self.K = len(self.beta)

    def _word_like(self, k, w):
        # (n_kw + eta) / (n_k + V*eta)
        return (self.n_kw[k][w] + self.eta) / (self.n_k[k] + self.V*self.eta)

    def _new_like(self, w):
        # prior predictive under symmetric Dir(eta): 1/V
        return 1.0 / self.V

    def _sample_topic(self, d, w):
        weights = []
        for k in range(self.K):
            prior = self.n_dk[d][k] + self.alpha * self.beta[k]
            like  = self._word_like(k, w)
            weights.append(prior * like)
        neww = self.alpha * self.beta_rest * self._new_like(w)
        weights.append(neww)
        Z = sum(weights) or 1.0
        r = random.random() * Z
        c = 0.0
        for idx, val in enumerate(weights):
            c += val
            if r <= c: return idx
        return self.K

    def _resample_beta(self):
        # m_k = sum_d m_dk, with m_dk ~ Antoniak(n_dk, alpha*beta_k)
        m_k = [0.0]*self.K
        for d in range(self.D):
            for k in range(self.K):
                ndk = self.n_dk[d].get(k, 0)
                if ndk > 0:
                    a = self.alpha * self.beta[k]
                    m_k[k] += self._antoniak_draw(ndk, a)
        # β ~ Dir(m_1,...,m_K, gamma)
        params = m_k + [self.gamma]
        full = self._dirichlet_sample(params)
        self.beta = full[:-1]
        self.beta_rest = full[-1]

    def log_likelihood(self):
        ll = 0.0
        for k in range(self.K):
            nk = self.n_k[k]
            ll += math.lgamma(self.V*self.eta) - math.lgamma(nk + self.V*self.eta)
            for w,c in self.n_kw[k].items():
                ll += math.lgamma(c + self.eta) - math.lgamma(self.eta)
        return ll

    def initialize(self):
        for d, doc in enumerate(self.corpus):
            for n, w in enumerate(doc):
                if self.K == 0: self._ensure_topic()
                choice = self._sample_topic(d, w)
                if choice == self.K: choice = self._ensure_topic()
                self.z[d][n] = choice
                self.n_dk[d][choice] += 1
                self.n_kw[choice][w] += 1
                self.n_k[choice] += 1
        self._maybe_compact()

    def iterate(self, iters=200, log_every=50):
        for it in range(1, iters+1):
            # decrement & resample
            for d, doc in enumerate(self.corpus):
                for n, w in enumerate(doc):
                    k = self.z[d][n]
                    if k != -1:
                        self.n_dk[d][k] -= 1;  self.n_kw[k][w] -= 1;  self.n_k[k] -= 1
                        if self.n_dk[d][k] == 0: del self.n_dk[d][k]
                        if self.n_kw[k][w] == 0: del self.n_kw[k][w]
                    self._maybe_compact()
                    choice = self._sample_topic(d, w)
                    if choice == self.K: choice = self._ensure_topic()
                    self.z[d][n] = choice
                    self.n_dk[d][choice] += 1; self.n_kw[choice][w] += 1; self.n_k[choice] += 1
            # resample beta
            self._resample_beta()
            if log_every and (it % log_every == 0 or it == 1):
                print(f"iter {it:3d} | K={self.K:2d} | ll={self.log_likelihood():.2f}")

    def top_words(self, k, topn=8):
        return [self.vocab[w] for w,_ in Counter(self.n_kw[k]).most_common(topn)]

    def print_summary(self, topn=8):
        print("\nDiscovered topics (top words):")
        for k in range(self.K):
            print(f"Topic {k:2d}:", self.top_words(k, topn))
        print("\nDoc-topic counts/proportions:")
        for d in range(self.D):
            row = [self.n_dk[d].get(k,0) for k in range(self.K)]
            s = sum(row) or 1
            props = [c/s for c in row]
            print(f"Doc {d:2d}:", DOCS_RAW[d])
            print("  counts:", row)
            print("  props :", [round(p,3) for p in props])

def main():
    docs = [tokenize(s) for s in DOCS_RAW]
    model = HDPLDADirectAssign(docs, eta=ETA, alpha=ALPHA, gamma=GAMMA, seed=SEED)
    print(f"Docs: {len(docs)}, Vocab: {model.V}, Tokens: {sum(len(d) for d in model.corpus)}")
    model.initialize()
    model.iterate(ITERS, log_every=50)
    model.print_summary(TOPN)

if __name__ == "__main__":
    main()
