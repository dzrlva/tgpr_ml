import os
import json
import subprocess
import tempfile
import numpy as np
from sklearn.cluster import KMeans


class RewardFunction:
    def __init__(self, config, corpus):
        self.w1, self.w2, self.w3 = config["reward"]["weights"]
        self.beta = config["reward"]["beta"]
        self.corpus = corpus
        self.corpus_features = None
        self.corpus_hist = None
        self._prepare_corpus()

    def _prepare_corpus(self):
        if self.corpus.embeddings is None:
            self.corpus.build_embeddings()
        self.corpus_features = self.corpus.embeddings
        self.corpus_hist = self.corpus.histogram

    def compute(self, code, group_codes):
        r1 = self._executability(code)
        r2 = self._distributional_pr(group_codes)
        r3 = self._gated_sc(code, group_codes, r2["precision"])
        return self.w1 * r1 + self.w2 * r2["f_beta"] + self.w3 * r3

    def _executability(self, code):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            path = f.name

        try:
            result = subprocess.run(
                ["python", path],
                capture_output=True,
                text=True,
                timeout=5
            )
            os.unlink(path)

            if result.returncode != 0:
                return 0.0
            if result.stderr and "warning" in result.stderr.lower():
                return 0.5
            return 1.0
        except Exception:
            if os.path.exists(path):
                os.unlink(path)
            return 0.0

    def _distributional_pr(self, group_codes):
        group_features = self.corpus.extract_features(group_codes)
        all_features = np.vstack([self.corpus_features, group_features])

        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        labels = kmeans.fit_predict(all_features)

        n_corpus = len(self.corpus_features)
        P = np.zeros(20)
        Q = np.zeros(20)

        for lab in labels[:n_corpus]:
            P[lab] += 1
        for lab in labels[n_corpus:]:
            Q[lab] += 1

        P = P / (P.sum() + 1e-8)
        Q = Q / (Q.sum() + 1e-8)

        precision = np.sum(np.minimum(P, Q))
        recall = precision
        f_beta = (
            (1 + self.beta**2) * precision * recall
            / (self.beta**2 * precision + recall + 1e-8)
        )

        return {"precision": precision, "recall": recall, "f_beta": f_beta}

    def _gated_sc(self, code, group_codes, precision):
        gate = max(0.0, 1.0 - 2.0 * abs(precision - 0.5))

        target = self._normalize(code)
        matches = sum(1 for c in group_codes if self._normalize(c) == target)
        sc = matches / len(group_codes)

        return gate * sc

    def _normalize(self, code):
        lines = [l.strip() for l in code.strip().split('\n') if l.strip()]
        return '\n'.join(lines)
