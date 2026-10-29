import os
import re
import math
import subprocess


def tokenize(text):
    return re.findall(r"[a-z0-9]+|[$!]", text.lower())


def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label, text = line.split(",", 1)
            rows.append((label.strip(), text.strip()))
    return rows


class NaiveBayes:
    def __init__(self):
        self.counts = {"spam": {}, "ham": {}}
        self.docs = {"spam": 0, "ham": 0}
        self.total_words = {"spam": 0, "ham": 0}
        self.vocab = set()

    def train(self, rows):
        for label, text in rows:
            self.docs[label] += 1
            for w in tokenize(text):
                self.counts[label][w] = self.counts[label].get(w, 0) + 1
                self.total_words[label] += 1
                self.vocab.add(w)

    def log_prob(self, word, label):
        # Laplace smoothing: add 1 to every count so no word has zero chance
        return math.log((self.counts[label].get(word, 0) + 1) / (self.total_words[label] + len(self.vocab)))

    def predict(self, text):
        words = [w for w in tokenize(text) if w in self.vocab]
        n = self.docs["spam"] + self.docs["ham"]
        score_spam = math.log(self.docs["spam"] / n)
        score_ham = math.log(self.docs["ham"] / n)
        push = {}
        for w in words:
            score_spam += self.log_prob(w, "spam")
            score_ham += self.log_prob(w, "ham")
            push[w] = self.log_prob(w, "spam") - self.log_prob(w, "ham")
        diff = max(-500.0, min(500.0, score_spam - score_ham))
        p_spam = 1.0 / (1.0 + math.exp(-diff))
        ranked = sorted(push.items(), key=lambda kv: kv[1], reverse=True)
        return p_spam, ranked


TESTS = [
    ("spam", "Congratulations! You won a free prize. Click here to claim now"),
    ("spam", "Urgent: verify your account password now"),
    ("ham", "Can you send me the report before the meeting"),
    ("ham", "Are you coming to lunch tomorrow"),
    ("spam", "Free offer just for you"),
    ("ham", "Please call me after the lecture"),
]

if __name__ == "__main__":
    rows = load_rows(os.path.join("data", "sample_messages.csv"))
    model = NaiveBayes()
    model.train(rows)

    lines = []
    lines.append("SPAM FILTER - STEP 1 REPORT")
    lines.append("-" * 60)
    lines.append("Training messages: spam = " + str(model.docs["spam"]) + ", ham = " + str(model.docs["ham"]))
    lines.append("Different words learned: " + str(len(model.vocab)))
    lines.append("Words in spam: " + str(model.total_words["spam"]) + ", words in ham: " + str(model.total_words["ham"]))
    lines.append("-" * 60)

    correct = 0
    for expected, text in TESTS:
        p, ranked = model.predict(text)
        guess = "spam" if p >= 0.5 else "ham"
        ok = guess == expected
        if ok:
            correct += 1
        lines.append("Message: " + text)
        lines.append("Expected: " + expected + " | Guess: " + guess + " | Spam chance: " + format(p * 100, ".1f") + "% | " + ("CORRECT" if ok else "WRONG"))
        top_spam = [w + " (" + format(v, "+.2f") + ")" for w, v in ranked[:3] if v > 0]
        top_ham = [w + " (" + format(v, "+.2f") + ")" for w, v in ranked[::-1][:3] if v < 0]
        lines.append("  Pushed toward spam: " + (", ".join(top_spam) if top_spam else "none"))
        lines.append("  Pushed toward ham: " + (", ".join(top_ham) if top_ham else "none"))
        lines.append("-" * 60)

    lines.append("Score: " + str(correct) + " of " + str(len(TESTS)) + " correct")

    out = os.path.abspath("step1_report.txt")
    with open(out, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    subprocess.Popen(["notepad.exe", out])
