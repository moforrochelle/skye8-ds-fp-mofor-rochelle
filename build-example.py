import bisect, csv
import xml.etree.ElementTree as ET
from collections import defaultdict

MEETINGS = ["ES2008a", "ES2008b"]
WORDS_DIR = "data/amicorpus/ami_public_manual_1/words"
PAUSES = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5]
RESUME_LIMIT = 2.0
FUNCTION = {"and", "but", "so", "the", "a", "to", "of", "or", "if", "that",
            "because", "i", "we", "you", "it", "is", "in", "for", "with"}
HESITATION = {"um", "uh", "mm", "hmm", "er", "erm"}


def load(meeting):
    words = []
    for spk in "ABCD":
        root = ET.parse(f"{WORDS_DIR}/{meeting}.{spk}.words.xml").getroot()
        for el in root:
            if el.tag != "w" or el.get("punc") == "true":
                continue
            if el.get("starttime") is None or el.get("endtime") is None:
                continue
            words.append((float(el.get("starttime")), float(el.get("endtime")),
                          spk, (el.text or "").strip().lower()))
    words.sort()
    return words


def real_turn_starts(ws):
    # a "real" start: gap of more than 0.5 s before it, then 3 words within 1.5 s
    starts = []
    for k in range(len(ws)):
        gap_before = k == 0 or ws[k][0] - ws[k - 1][1] > 0.5
        three_words = k + 2 < len(ws) and ws[k + 2][0] - ws[k][0] < 1.5
        if gap_before and three_words:
            starts.append(ws[k][0])
    return starts


rows = []
for m in MEETINGS:
    words = load(m)
    by_spk = defaultdict(list)
    for w in words:
        by_spk[w[2]].append(w)
    starts_by_spk = {s: real_turn_starts(ws) for s, ws in by_spk.items()}
    for spk, ws in by_spk.items():
        other_starts = sorted(
            t for o, ts in starts_by_spk.items() if o != spk for t in ts
        )
        n_utt = 0
        for i in range(len(ws) - 1):
            start, end, _, text = ws[i]
            if i > 0 and start - ws[i - 1][1] > 0.5:
                n_utt = 0
            n_utt += 1
            next_s = ws[i + 1][0]
            j = bisect.bisect_right(other_starts, end)
            next_o = other_starts[j] if j < len(other_starts) else float("inf")
            gap = next_s - end
            ended = 1 if (next_o < next_s or gap > RESUME_LIMIT) else 0
            for p in PAUSES:
                if gap >= p:
                    rows.append([m, spk, p, int(text in FUNCTION),
                                 int(text in HESITATION), len(text),
                                 min(n_utt, 40), ended])

with open("data/endpoint_examples.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["meeting", "speaker", "pause", "last_is_function",
                 "last_is_hesitation", "last_len", "n_words_utt", "ended"])
    wr.writerows(rows)

total = len(rows)
ended = sum(r[-1] for r in rows)
print("total examples:", total)
print("inside-turn (hold):", total - ended)
print("end-of-turn:", ended)
for p in PAUSES:
    sub = [r for r in rows if r[2] == p]
    print(f"pause {p}s: {len(sub)} examples, {sum(r[-1] for r in sub)} end-of-turn")