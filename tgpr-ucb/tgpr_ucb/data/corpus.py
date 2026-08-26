import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import torch


class ValidatedCorpus:
    def __init__(self, path):
        self.path = path
        self.samples = []
        self.embeddings = None
        self.histogram = None

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.encoder.eval()

    def load(self):
        with open(self.path) as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data.get("code", data.get("content", "")))

    def extract_features(self, codes):
        features = []
        with torch.no_grad():
            for code in codes:
                inputs = self.tokenizer(
                    code,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding="max_length"
                )
                outputs = self.encoder(**inputs)
                vec = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                features.append(vec)
        return np.array(features)

    def build_embeddings(self):
        if not self.samples:
            self.load()
        self.embeddings = self.extract_features(self.samples)

        kmeans = KMeans(n_clusters=20, random_state=42, n_init=10)
        labels = kmeans.fit_predict(self.embeddings)

        hist = np.zeros(20)
        for lab in labels:
            hist[lab] += 1
        self.histogram = hist / (hist.sum() + 1e-8)
