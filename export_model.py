import csv, json
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

FEATS = ["pause", "last_is_function", "last_is_hesitation", "last_len", "n_words_utt"]
FUNCTION = ["and", "but", "so", "the", "a", "to", "of", "or", "if", "that",
            "because", "i", "we", "you", "it", "is", "in", "for", "with"]
HESITATION = ["um", "uh", "mm", "hmm", "er", "erm"]
THRESHOLD = 0.98

rows = list(csv.DictReader(open("data/endpoint_examples.csv")))
for r in rows:
    for k in FEATS + ["ended"]:
        r[k] = float(r[k])
train = [r for r in rows if r["meeting"] == "ES2008a"]
test = [r for r in rows if r["meeting"] == "ES2008b"]


def X(rs): return np.array([[r[k] for k in FEATS] for r in rs])
def y(rs): return np.array([int(r["ended"]) for r in rs])


model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
model.fit(X(train), y(train))
lr = model.learning_rate

trees = []
for est in model.estimators_[:, 0]:
    t = est.tree_
    trees.append({
        "left": t.children_left.tolist(),
        "right": t.children_right.tolist(),
        "feature": t.feature.tolist(),
        "threshold": t.threshold.tolist(),
        "value": t.value[:, 0, 0].tolist(),
    })


def tree_sum(x):
    s = 0.0
    for t in trees:
        n = 0
        while t["left"][n] != -1:
            n = t["left"][n] if x[t["feature"][n]] <= t["threshold"][n] else t["right"][n]
        s += t["value"][n]
    return s


Xtr = X(train)
base = float(model.decision_function(Xtr[:1])[0]) - lr * tree_sum(Xtr[0])

Xte = X(test)
mine = np.array([1 / (1 + np.exp(-(base + lr * tree_sum(x)))) for x in Xte])
theirs = model.predict_proba(Xte)[:, 1]
print("trees:", len(trees))
print("max difference between my evaluator and sklearn:", float(np.max(np.abs(mine - theirs))))

json.dump({
    "features": FEATS,
    "base": base,
    "learning_rate": lr,
    "trees": trees,
    "threshold": THRESHOLD,
    "function_words": FUNCTION,
    "hesitations": HESITATION,
}, open("results/endpoint_model.json", "w"))
print("Saved results/endpoint_model.json")