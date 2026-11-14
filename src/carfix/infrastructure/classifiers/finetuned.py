"""Fine-tuned classifier: DistilBERT + a classification head, trained on our labelled examples.

Same recipe as the Session 5.2 lab (Part B), written as a plain PyTorch loop so every step is visible:
tokenise -> forward -> cross-entropy loss -> backward -> AdamW step, a few epochs, keep the best epoch on the
VALIDATION split (never on the test split).

`untrained=True` gives the "base model": the same pretrained DistilBERT with a fresh, untrained head. It is
the literal "base model" baseline the assignment asks for (expected: close to chance).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from carfix.domain.models import LabelledText


def _device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class FineTunedClassifier:
    def __init__(self, model_name_or_path: str, labels: list[str] | None = None, max_length: int = 64,
                 device: str | None = None, name: str | None = None):
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

        path = Path(model_name_or_path)
        if labels is None:  # loading a saved model (local folder or Hub id): labels are in its config
            id2label = AutoConfig.from_pretrained(model_name_or_path).id2label
            labels = [id2label[i] for i in range(len(id2label))]
        self.labels = labels
        self.max_length = max_length
        self.device = device or _device()
        self.name = name or f"fine-tuned/{path.name}"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id={l: i for i, l in enumerate(labels)},
        ).to(self.device)

    # ------------------------------------------------------------- training

    def fit(self, train: list[LabelledText], val: list[LabelledText], epochs: int = 4, lr: float = 5e-5,
            batch_size: int = 16, seed: int = 42, log=print) -> list[dict]:
        import torch
        from torch.optim import AdamW
        from transformers import get_linear_schedule_with_warmup

        set_seed(seed)
        y = {l: i for i, l in enumerate(self.labels)}
        opt = AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        steps = epochs * ((len(train) + batch_size - 1) // batch_size)
        sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

        history, best_acc, best_state = [], -1.0, None
        for epoch in range(1, epochs + 1):
            self.model.train()
            order = list(range(len(train)))
            random.shuffle(order)
            total = 0.0
            for i in range(0, len(order), batch_size):
                batch = [train[j] for j in order[i:i + batch_size]]
                enc = self._encode([e.text for e in batch])
                labels = torch.tensor([y[e.label] for e in batch], device=self.device)
                loss = self.model(**enc, labels=labels).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad()
                total += loss.item() * len(batch)
            val_acc = float(np.mean([p == e.label for p, e in zip(self.predict([e.text for e in val]), val)]))
            history.append({"epoch": epoch, "train_loss": total / len(train), "val_accuracy": val_acc})
            log(f"  epoch {epoch}: train loss {total / len(train):.4f}, val accuracy {val_acc:.3f}")
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        return history

    # ------------------------------------------------------------- inference

    def _encode(self, texts: list[str]):
        enc = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        import torch
        self.model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                logits = self.model(**self._encode(texts[i:i + batch_size])).logits
                out.append(torch.softmax(logits, dim=-1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.labels)))

    def predict(self, texts: list[str]) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(texts).argmax(axis=1)]

    def save(self, directory: str | Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
