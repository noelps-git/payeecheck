"""
train_l5.py — Fine-tune a Siamese sentence-transformer on Indian name pairs.

HONEST NOTE ON DATA:
The training pairs below are illustrative examples from the PayeeCheck
Engineering Playbook — enough to make the training loop run end-to-end
and produce a working model file. They are NOT enough data to meaningfully
beat Level 4 in production.

To get real accuracy gains over Level 4, replace `train_examples` below
with pairs derived from:
  1. Your own KYC records — same customer, different name spellings
  2. A real MCA company list (see matchers/corpus.py for download instructions)
     — generate synthetic positive pairs by adding/removing entity suffixes,
       reordering tokens, and applying common transliteration substitutions
  3. Negative pairs — random cross-pairs from the same corpus

Rule of thumb from the playbook: you need 5,000-10,000+ labelled pairs
before Level 5 reliably beats Level 4. Run benchmark.py after training to
see honestly whether your fine-tune actually improved anything.
"""
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator
from torch.utils.data import DataLoader
import os

# ── TRAINING DATA — illustrative seed set, see module docstring ──────
train_examples = [
    InputExample(texts=["State Bank of India", "SBI"], label=1.0),
    InputExample(texts=["HDFC Bank", "HDFC Bank Limited"], label=1.0),
    InputExample(texts=["Mohammed Riaz", "Mohammad Riaz"], label=1.0),
    InputExample(texts=["Suresh Kumar", "Suresh Kumar Pvt Ltd"], label=0.6),
    InputExample(texts=["Krishna Enterprises", "Ramesh D"], label=0.0),
    InputExample(texts=["Amazon Pay", "Amazon Pay India"], label=1.0),
    InputExample(texts=["Reliance Jio", "Reliance Industries"], label=0.3),
    InputExample(texts=["Paytm", "Paytm Payments Bank"], label=0.8),
    InputExample(texts=["ICICI Bank", "Punjab National Bank"], label=0.0),
    InputExample(texts=["Muthu Kumar", "Muthukumar S"], label=0.9),
    InputExample(texts=["HDFC Bank", "HDFC Life"], label=0.2),
    InputExample(texts=["Kumar Suresh", "Suresh Kumar"], label=1.0),
    InputExample(texts=["PNB", "Punjab National Bank"], label=1.0),
    InputExample(texts=["BOB", "Bank of Baroda"], label=1.0),
    InputExample(texts=["Axis Bank", "Wipro Limited"], label=0.0),
]

# ── EVALUATION DATA ───────────────────────────────────────────────────
eval_s1 = ["State Bank of India", "Amazon Pay", "Mohammed Riaz", "HDFC Bank"]
eval_s2 = ["SBI", "Amazon Pay India", "Mohammad Riaz K", "HDFC Life"]
eval_labels = [1, 1, 1, 0]
evaluator = BinaryClassificationEvaluator(eval_s1, eval_s2, eval_labels)


def train(output_path="payeecheck_name_model", epochs=10):
    print(f"Training on {len(train_examples)} labelled pairs "
          f"(illustrative seed set — see module docstring for how to scale this)")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    train_dl = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.CosineSimilarityLoss(model)

    model.fit(
        train_objectives=[(train_dl, train_loss)],
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=5,
        output_path=output_path,
        save_best_model=True,
    )
    print(f"\nModel saved to ./{output_path}")
    print("Run `python -m matchers.l5_siamese` to test it, "
          "or `python tests/benchmark.py` to compare against all other levels.")


if __name__ == "__main__":
    train()
