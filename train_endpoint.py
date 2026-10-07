import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier

FEATS = ["pause", "last_is_function", "last_is_hesitation", "last_len", "n_words_utt"]
PAUSES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
FALLBACK = 2.0      # hard timeout if the model never says "finished"
CUT_LIMIT = 0.20    # operating point: at most 20% of thinking pauses interrupted

rows = list(csv.DictReader(open("data/endpoint_examples.csv")))
for r in rows:
    for k in FEATS + ["ended"]:
        r[k] = float(r[k])

# each pause event starts at pause = 0.2
event = -1
for r in rows:
    if r["pause"] == 0.2:
        event += 1
    r["event"] = event

train = [r for r in rows if r["meeting"] == "ES2008a"]
test = [r for r in rows if r["meeting"] == "ES2008b"]


def X(rs): return np.array([[r[k] for k in FEATS] for r in rs])
def y(rs): return np.array([int(r["ended"]) for r in rs])


model = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0)
model.fit(X(train), y(train))
prob = model.predict_proba(X(test))[:, 1]

events = {}
for r, p in zip(test, prob):
    events.setdefault(r["event"], []).append((r["pause"], p, int(r["ended"])))

n_hold = sum(1 for e in events.values() if e[0][2] == 0)
n_end = len(events) - n_hold
print(f"train rows: {len(train)} | test events: {len(events)} "
      f"({n_hold} thinking pauses, {n_end} finished turns)")


def run_model(thr):
    cuts, waits = 0, []
    for evs in events.values():
        fired = next((pz for pz, p, _ in evs if p >= thr), None)
        if evs[0][2]:
            waits.append(fired if fired is not None else FALLBACK)
        elif fired is not None:
            cuts += 1
    return cuts / n_hold, float(np.mean(waits))


def run_timer(T):
    cuts, waits = 0, []
    for evs in events.values():
        if evs[0][2]:
            waits.append(T)
        elif any(pz >= T for pz, _, _ in evs):
            cuts += 1
    return cuts / n_hold, float(np.mean(waits))


thrs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99]
model_pts = [(t,) + run_model(t) for t in thrs]
timer_pts = [(T,) + run_timer(T) for T in PAUSES]

print("\nMODEL (threshold, cut-off rate, mean wait s)")
for t, c, w in model_pts:
    print(f"  {t:.2f}  {c*100:5.1f}%  {w:.2f}")
print("\nFIXED TIMER (seconds, cut-off rate, mean wait s)")
for t, c, w in timer_pts:
    print(f"  {t:.1f}  {c*100:5.1f}%  {w:.2f}")


def best(pts):
    ok = [p for p in pts if p[1] <= CUT_LIMIT]
    return min(ok, key=lambda p: p[2]) if ok else None


bm, bt = best(model_pts), best(timer_pts)
print(f"\nOperating point rule: at most {CUT_LIMIT*100:.0f}% cut-offs, then lowest wait")
print("  model:", None if bm is None else f"threshold {bm[0]}, cut-offs {bm[1]*100:.1f}%, wait {bm[2]:.2f}s")
print("  timer:", None if bt is None else f"{bt[0]}s, cut-offs {bt[1]*100:.1f}%, wait {bt[2]:.2f}s")

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot([p[2] for p in model_pts], [p[1] * 100 for p in model_pts], "o-", label="trained model")
plt.plot([p[2] for p in timer_pts], [p[1] * 100 for p in timer_pts], "s--", label="fixed silence timer")
if bm:
    plt.scatter([bm[2]], [bm[1] * 100], s=160, facecolors="none", edgecolors="red", label="chosen operating point")
plt.xlabel("Mean wait after speaker has finished (s)")
plt.ylabel("Thinking pauses interrupted (%)")
plt.legend()
plt.tight_layout()
plt.savefig("results/tradeoff.png", dpi=150)
print("\nSaved results/tradeoff.png")