import csv, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATS = ["pause", "last_is_function", "last_is_hesitation", "last_len", "n_words_utt"]
FUNCTION = ["and", "but", "so", "the", "a", "to", "of", "or", "if", "that",
            "because", "i", "we", "you", "it", "is", "in", "for", "with"]
HESITATION = ["um", "uh", "mm", "hmm", "er", "erm"]
FALLBACK = 2.0
CUT_LIMIT = 0.20

rows = list(csv.DictReader(open("data/endpoint_examples.csv")))
for r in rows:
    for k in FEATS + ["ended"]:
        r[k] = float(r[k])

event = -1
for r in rows:
    if r["pause"] == 0.2:
        event += 1
    r["event"] = event

train = [r for r in rows if r["meeting"] == "ES2008a"]
test = [r for r in rows if r["meeting"] == "ES2008b"]


def X(rs): return np.array([[r[k] for k in FEATS] for r in rs])
def y(rs): return np.array([int(r["ended"]) for r in rs])


scaler = StandardScaler().fit(X(train))
clf = LogisticRegression(max_iter=1000).fit(scaler.transform(X(train)), y(train))
prob = clf.predict_proba(scaler.transform(X(test)))[:, 1]

events = {}
for r, p in zip(test, prob):
    events.setdefault(r["event"], []).append((r["pause"], p, int(r["ended"])))
n_hold = sum(1 for e in events.values() if e[0][2] == 0)


def run_model(thr):
    cuts, waits = 0, []
    for evs in events.values():
        fired = next((pz for pz, p, _ in evs if p >= thr), None)
        if evs[0][2]:
            waits.append(fired if fired is not None else FALLBACK)
        elif fired is not None:
            cuts += 1
    return cuts / n_hold, float(np.mean(waits))


thrs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
pts = [(t,) + run_model(t) for t in thrs]
print("LOGISTIC REGRESSION (threshold, cut-off rate, mean wait s)")
for t, c, w in pts:
    print(f"  {t:.2f}  {c*100:5.1f}%  {w:.2f}")

ok = [p for p in pts if p[1] <= CUT_LIMIT]
chosen = min(ok, key=lambda p: p[2]) if ok else min(pts, key=lambda p: p[1])
print(f"\nChosen: threshold {chosen[0]}, cut-offs {chosen[1]*100:.1f}%, wait {chosen[2]:.2f}s")
print("Weights:", dict(zip(FEATS, np.round(clf.coef_[0], 3))))

os.makedirs("results", exist_ok=True)
out = {
    "features": FEATS,
    "mean": scaler.mean_.tolist(),
    "scale": scaler.scale_.tolist(),
    "coef": clf.coef_[0].tolist(),
    "intercept": float(clf.intercept_[0]),
    "threshold": chosen[0],
    "function_words": FUNCTION,
    "hesitations": HESITATION,
}
json.dump(out, open("results/endpoint_weights.json", "w"), indent=2)
print("Saved results/endpoint_weights.json")