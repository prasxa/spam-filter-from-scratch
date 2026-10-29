import os
import json
import random
import zipfile
import traceback
import subprocess
import urllib.request

from classifier import NaiveBayes, tokenize

URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
ZIP_PATH = os.path.join("data", "sms_spam_collection.zip")
TSV_PATH = os.path.join("data", "SMSSpamCollection.txt")


def get_data():
    if not os.path.exists(TSV_PATH):
        if not os.path.exists(ZIP_PATH):
            req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as resp:
                with open(ZIP_PATH, "wb") as f:
                    f.write(resp.read())
        with zipfile.ZipFile(ZIP_PATH) as z:
            for name in z.namelist():
                if name.lower() == "smsspamcollection":
                    with open(TSV_PATH, "wb") as f:
                        f.write(z.read(name))
    rows = []
    with open(TSV_PATH, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if "\t" not in line:
                continue
            label, text = line.split("\t", 1)
            if label in ("spam", "ham"):
                rows.append((label, text.strip()))
    return rows


def save_model(model, path):
    data = {
        "counts": model.counts,
        "docs": model.docs,
        "total_words": model.total_words,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


def main():
    lines = []
    rows = get_data()
    random.Random(42).shuffle(rows)
    cut = int(len(rows) * 0.8)
    train_rows = rows[:cut]
    test_rows = rows[cut:]

    model = NaiveBayes()
    model.train(train_rows)

    tp = fp = tn = fn = 0
    fp_examples = []
    fn_examples = []
    for label, text in test_rows:
        p, _ = model.predict(text)
        guess = "spam" if p >= 0.5 else "ham"
        if label == "spam" and guess == "spam":
            tp += 1
        elif label == "ham" and guess == "spam":
            fp += 1
            if len(fp_examples) < 5:
                fp_examples.append(text[:100])
        elif label == "ham" and guess == "ham":
            tn += 1
        else:
            fn += 1
            if len(fn_examples) < 5:
                fn_examples.append(text[:100])

    total = len(test_rows)
    accuracy = (tp + tn) / total * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) else 0.0
    recall = tp / (tp + fn) * 100 if (tp + fn) else 0.0

    lines.append("SPAM FILTER - STEP 2 REPORT")
    lines.append("-" * 60)
    lines.append("Total real messages: " + str(len(rows)))
    lines.append("Training messages: " + str(len(train_rows)) + " | Test messages (never seen): " + str(len(test_rows)))
    lines.append("Different words learned: " + str(len(model.vocab)))
    lines.append("-" * 60)
    lines.append("RESULTS ON THE UNSEEN TEST MESSAGES")
    lines.append("Spam caught correctly (TP): " + str(tp))
    lines.append("Spam missed (FN): " + str(fn))
    lines.append("Normal flagged as spam by mistake (FP): " + str(fp))
    lines.append("Normal passed correctly (TN): " + str(tn))
    lines.append("Accuracy: " + format(accuracy, ".1f") + "%")
    lines.append("Precision (of messages flagged spam, how many really were): " + format(precision, ".1f") + "%")
    lines.append("Recall (of all real spam, how much was caught): " + format(recall, ".1f") + "%")
    lines.append("-" * 60)

    ratios = []
    for w in model.vocab:
        seen = model.counts["spam"].get(w, 0) + model.counts["ham"].get(w, 0)
        if seen >= 10:
            ratios.append((w, model.log_prob(w, "spam") - model.log_prob(w, "ham")))
    ratios.sort(key=lambda kv: kv[1], reverse=True)
    lines.append("TOP 10 SPAM WORDS (seen 10+ times)")
    lines.append(", ".join(w + " (" + format(v, "+.2f") + ")" for w, v in ratios[:10]))
    lines.append("TOP 10 NORMAL WORDS (seen 10+ times)")
    lines.append(", ".join(w + " (" + format(v, "+.2f") + ")" for w, v in ratios[::-1][:10]))
    lines.append("-" * 60)

    lines.append("NORMAL MESSAGES WRONGLY FLAGGED AS SPAM (up to 5)")
    for t in fp_examples:
        lines.append("  " + t)
    lines.append("SPAM THAT SLIPPED THROUGH (up to 5)")
    for t in fn_examples:
        lines.append("  " + t)
    lines.append("-" * 60)

    # Final model for the web page: trained on ALL messages
    full = NaiveBayes()
    full.train(rows)
    save_model(full, "model.json")
    lines.append("Saved model.json (trained on all " + str(len(rows)) + " messages) - size: " + str(os.path.getsize("model.json")) + " bytes")

    return lines


if __name__ == "__main__":
    try:
        report_lines = main()
    except Exception:
        report_lines = ["STEP 2 FAILED", "-" * 60, traceback.format_exc(),
                        "-" * 60,
                        "If this is a download problem, the internet or a firewall blocked the file."]
    out = os.path.abspath("step2_report.txt")
    with open(out, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(report_lines))
    subprocess.Popen(["notepad.exe", out])
