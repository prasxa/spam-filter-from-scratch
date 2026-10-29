import os
import traceback
import subprocess

from app import app

lines = []
try:
    client = app.test_client()
    r = client.get("/")
    lines.append("GET / -> status " + str(r.status_code) + ", page size " + str(len(r.data)) + " bytes")
    r = client.post("/api/check", json={"text": "   "})
    lines.append("POST empty text -> status " + str(r.status_code) + " (expected 400)")
    lines.append("-" * 60)
    tests = [
        ("spam", "WINNER! You have been selected to receive a $900 prize reward. To claim call now"),
        ("spam", "Your account has been locked. Click here to verify your password now"),
        ("ham", "Hey are you coming for dinner tonight? Mom said she will cook at 8"),
        ("ham", "Can you send me the notes from the lecture"),
        ("spam", "Free entry in a contest. Text WIN to 80085 to claim your prize"),
        ("ham", "I will be late. Start the meeting without me"),
    ]
    correct = 0
    for expected, text in tests:
        r = client.post("/api/check", json={"text": text})
        d = r.get_json()
        guess = "spam" if d["percent"] >= 50 else "ham"
        ok = guess == expected
        if ok:
            correct += 1
        lines.append("Message: " + text)
        lines.append("Expected " + expected + " | Spam chance " + str(d["percent"]) + "% | " + d["verdict"] + " | " + ("CORRECT" if ok else "WRONG"))
        lines.append("  Top spam words: " + ", ".join(x["word"] for x in d["top_spam"]))
        lines.append("  Top normal words: " + ", ".join(x["word"] for x in d["top_ham"]))
        lines.append("-" * 60)
    lines.append("Score: " + str(correct) + " of " + str(len(tests)) + " correct")
except Exception:
    lines.append("SELF-TEST FAILED")
    lines.append(traceback.format_exc())

out = os.path.abspath("step3_report.txt")
with open(out, "w", encoding="utf-8-sig") as f:
    f.write("\n".join(lines))
subprocess.Popen(["notepad.exe", out])
