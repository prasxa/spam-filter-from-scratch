import json
import math

from flask import Flask, request, jsonify, render_template_string

from classifier import NaiveBayes, tokenize

app = Flask(__name__)


def load_model():
    m = NaiveBayes()
    with open("model.json", "r", encoding="utf-8") as f:
        d = json.load(f)
    m.counts = d["counts"]
    m.docs = d["docs"]
    m.total_words = d["total_words"]
    m.vocab = set(m.counts["spam"]) | set(m.counts["ham"])
    return m


MODEL = load_model()


def verdict_for(p):
    if p >= 0.9:
        return "Very likely spam"
    if p >= 0.5:
        return "Probably spam"
    if p >= 0.1:
        return "Probably normal"
    return "Very likely normal"


def check(text):
    tokens = tokenize(text)
    n = MODEL.docs["spam"] + MODEL.docs["ham"]
    score_spam = math.log(MODEL.docs["spam"] / n)
    score_ham = math.log(MODEL.docs["ham"] / n)
    token_info = []
    seen = {}
    unknown = 0
    for w in tokens:
        if w in MODEL.vocab:
            ls = MODEL.log_prob(w, "spam")
            lh = MODEL.log_prob(w, "ham")
            score_spam += ls
            score_ham += lh
            push = ls - lh
            token_info.append({"w": w, "push": round(push, 2)})
            seen[w] = push
        else:
            unknown += 1
            token_info.append({"w": w, "push": None})
    diff = max(-500.0, min(500.0, score_spam - score_ham))
    p = 1.0 / (1.0 + math.exp(-diff))
    ranked = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)
    top_spam = [{"word": w, "push": round(v, 2)} for w, v in ranked if v > 0][:5]
    top_ham = [{"word": w, "push": round(v, 2)} for w, v in ranked[::-1] if v < 0][:5]
    return {
        "percent": round(p * 100, 1),
        "verdict": verdict_for(p),
        "tokens": token_info,
        "top_spam": top_spam,
        "top_ham": top_ham,
        "unknown": unknown,
    }


@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", ""))[:5000]
    if not text.strip():
        return jsonify({"error": "Please paste a message first."}), 400
    return jsonify(check(text))


PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Spam Filter from Scratch</title>
<style>
  body { font-family: Segoe UI, Arial, sans-serif; background: #f4f6fb; color: #1d2433; margin: 0; }
  .wrap { max-width: 820px; margin: 0 auto; padding: 24px 16px 60px; }
  h1 { margin: 0 0 4px; font-size: 26px; }
  .sub { color: #5b6477; margin: 0 0 20px; }
  .card { background: #fff; border-radius: 12px; padding: 18px; box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 16px; }
  textarea { width: 100%; box-sizing: border-box; height: 120px; font-size: 15px; padding: 10px; border: 1px solid #c8cfdd; border-radius: 8px; font-family: inherit; }
  button { background: #2f5bea; color: #fff; border: 0; border-radius: 8px; padding: 10px 18px; font-size: 15px; cursor: pointer; }
  button.ghost { background: #e8ecf8; color: #2f5bea; padding: 6px 12px; font-size: 13px; margin-right: 6px; }
  .row { margin-top: 10px; }
  .pct { font-size: 44px; font-weight: 700; }
  .bar { height: 14px; background: #e3e7f1; border-radius: 7px; overflow: hidden; margin: 8px 0; }
  .fill { height: 100%; width: 0; background: #2f5bea; }
  .verdict { font-size: 18px; font-weight: 600; }
  .tk { display: inline-block; padding: 2px 6px; margin: 2px; border-radius: 6px; font-size: 15px; }
  .none { background: #eef0f5; color: #7b8396; }
  .mid { background: #f4f4f4; }
  .s1 { background: #ffd9d4; }
  .s2 { background: #ff9d92; }
  .h1 { background: #d7f0dc; }
  .h2 { background: #9fdcab; }
  .cols { display: flex; gap: 16px; flex-wrap: wrap; }
  .cols > div { flex: 1; min-width: 220px; }
  ul { margin: 6px 0 0; padding-left: 20px; }
  .small { color: #5b6477; font-size: 13px; }
  .err { color: #b3261e; margin-top: 8px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Spam Filter from Scratch</h1>
  <p class="sub">A Naive Bayes filter written in plain Python. Paste a message and see exactly which words pushed the score.</p>

  <div class="card">
    <textarea id="msg" placeholder="Paste a message here..."></textarea>
    <div class="row">
      <button id="go">Check message</button>
      <span class="small" style="margin-left:10px">Try an example:</span>
      <span id="examples"></span>
    </div>
    <div id="err" class="err"></div>
  </div>

  <div class="card" id="result" style="display:none">
    <div class="pct"><span id="pct">0</span>% spam chance</div>
    <div class="bar"><div class="fill" id="fill"></div></div>
    <div class="verdict" id="verdict"></div>
    <p class="small">How the filter saw your message (red = pushes toward spam, green = pushes toward normal, grey = word it never learned):</p>
    <div id="tokens"></div>
    <div class="cols row">
      <div><b>Strongest spam words</b><ul id="tspam"></ul></div>
      <div><b>Strongest normal words</b><ul id="tham"></ul></div>
    </div>
    <p class="small" id="unk"></p>
  </div>

  <div class="card small">
    <b>How well does it work?</b> Trained on 4,459 real SMS messages and tested on 1,115 it had never seen:
    98.3% accuracy, 91.2% of spam caught (145 of 159), and 5 of 956 normal messages wrongly flagged.
    It learned from SMS, so long emails may be less accurate. The number next to each word is the log
    ratio log P(word | spam) - log P(word | normal).
  </div>
</div>

<script>
const EXAMPLES = [
  ["Spam", "WINNER! You have been selected to receive a $900 prize reward. To claim call now, offer ends today"],
  ["Normal", "Hey are you coming for dinner tonight? Mom said she will cook at 8"],
  ["Phishing", "Your account has been locked. Click here to verify your password now"]
];

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function tokenClass(push) {
  if (push === null) return "tk none";
  if (push >= 1.5) return "tk s2";
  if (push >= 0.4) return "tk s1";
  if (push <= -1.5) return "tk h2";
  if (push <= -0.4) return "tk h1";
  return "tk mid";
}

function fillList(id, items, sign) {
  const ul = document.getElementById(id);
  ul.textContent = "";
  if (items.length === 0) {
    ul.appendChild(el("li", "", "none"));
    return;
  }
  items.forEach(function (it) {
    const s = (it.push > 0 ? "+" : "") + it.push.toFixed(2);
    ul.appendChild(el("li", "", it.word + " (" + s + ")"));
  });
}

function render(d) {
  document.getElementById("result").style.display = "block";
  document.getElementById("pct").textContent = d.percent.toFixed(1);
  document.getElementById("fill").style.width = d.percent + "%";
  document.getElementById("fill").style.background = d.percent >= 50 ? "#d93025" : "#1e8e3e";
  document.getElementById("verdict").textContent = d.verdict;
  const box = document.getElementById("tokens");
  box.textContent = "";
  d.tokens.forEach(function (t) {
    const s = el("span", tokenClass(t.push), t.w);
    s.title = t.push === null ? "not in training data" : "push: " + t.push.toFixed(2);
    box.appendChild(s);
  });
  fillList("tspam", d.top_spam);
  fillList("tham", d.top_ham);
  document.getElementById("unk").textContent = d.unknown + " word(s) were not in the training data and were ignored.";
}

async function runCheck() {
  const text = document.getElementById("msg").value;
  const err = document.getElementById("err");
  err.textContent = "";
  try {
    const r = await fetch("/api/check", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: text})
    });
    const d = await r.json();
    if (!r.ok) {
      err.textContent = d.error || "Something went wrong.";
      return;
    }
    render(d);
  } catch (e) {
    err.textContent = "Could not reach the server.";
  }
}

document.getElementById("go").addEventListener("click", runCheck);
EXAMPLES.forEach(function (ex) {
  const b = el("button", "ghost", ex[0]);
  b.addEventListener("click", function () {
    document.getElementById("msg").value = ex[1];
    runCheck();
  });
  document.getElementById("examples").appendChild(b);
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
