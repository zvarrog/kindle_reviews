import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoTokenizer, AutoModel
import torch


class DistilBertClassifier(BaseEstimator, ClassifierMixin):
    """Лёгкая обёртка DistilBERT: замороженный encoder + линейная голова."""

    def __init__(
        self,
        epochs=1,
        lr=1e-4,
        max_len=128,
        batch_size=8,
        seed=42,
        use_bigrams=False,
        device=None,
    ):
        self.epochs = epochs
        self.lr = lr
        self.max_len = max_len
        self.batch_size = batch_size
        self.seed = seed
        self.use_bigrams = use_bigrams
        self.device = device
        self._fitted = False
        self._tokenizer = None
        self._base_model = None
        self._head = None
        self._device_actual = None
        self._classes_ = None

    @staticmethod
    def _augment_texts(texts):
        def _augment_bigrams(s: str) -> str:
            ws = s.split()
            bigrams = [f"{ws[i]}_{ws[i+1]}" for i in range(len(ws) - 1)]
            return s + (" " + " ".join(bigrams[:20]) if bigrams else "")

        return np.array([_augment_bigrams(t) for t in texts])

    def fit(self, X, y):
        torch.manual_seed(self.seed)
        texts = X if isinstance(X, (list, np.ndarray)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        self._tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        self._base_model = AutoModel.from_pretrained("distilbert-base-uncased")
        device_str = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device_str)
        for p in self._base_model.parameters():
            p.requires_grad = False
        unique_labels = np.unique(y)
        self._classes_ = unique_labels
        label2idx = {lab: i for i, lab in enumerate(unique_labels)}
        hidden = self._base_model.config.hidden_size
        n_classes = len(unique_labels)
        self._head = torch.nn.Linear(hidden, n_classes).to(device)
        self._base_model.to(device)
        optimizer = torch.optim.Adam(self._head.parameters(), lr=self.lr)
        loss_fn = torch.nn.CrossEntropyLoss()
        self._base_model.eval()
        self._head.train()

        def batch_iter():
            for i in range(0, len(texts), self.batch_size):
                yield texts[i : i + self.batch_size], y[i : i + self.batch_size]

        for _ in range(self.epochs):
            for bt, by_raw in batch_iter():
                by = np.vectorize(label2idx.get)(by_raw).astype(int)
                enc = self._tokenizer(
                    list(bt),
                    truncation=True,
                    padding=True,
                    max_length=self.max_len,
                    clean_up_tokenization_spaces=False,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    out = self._base_model(**enc)
                    cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                loss = loss_fn(
                    logits, torch.tensor(by, dtype=torch.long, device=device)
                )
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self._device_actual = device
        self._head.eval()
        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("DistilBertClassifier не обучен")
        texts = X if isinstance(X, (list, np.ndarray)) else X.values
        if self.use_bigrams:
            texts = self._augment_texts(texts)
        preds = []
        device = self._device_actual
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            enc = self._tokenizer(
                list(batch),
                truncation=True,
                padding=True,
                max_length=self.max_len,
                clean_up_tokenization_spaces=False,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.no_grad():
                out = self._base_model(**enc)
                cls = out.last_hidden_state[:, 0]
                logits = self._head(cls)
                preds.extend(logits.argmax(dim=1).cpu().numpy())
        preds = np.array(preds)
        return self._classes_[preds]
