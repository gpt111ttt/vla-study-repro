import json
import os

R = "/home/vstc/openvla_probe/runs"


def load(tag, t):
    with open(os.path.join(R, tag, "eval_openvla_t%d_q1.json" % t)) as f:
        return json.load(f)


bad_s = bad_l = n = 0
len_pairs = []
for t in range(10):
    a, b = load("n1_r0", t), load("n1_r1_r1", t)
    ra = {r["episode"]: r for r in a["rows"]}
    rb = {r["episode"]: r for r in b["rows"]}
    if set(ra) != set(rb):
        print("task %d: episode sets differ" % t)
    for e in sorted(ra):
        n += 1
        if ra[e]["success"] != rb[e]["success"]:
            bad_s += 1
            print("  FLAG t%d e%d  %s vs %s" % (t, e, ra[e]["success"], rb[e]["success"]))
        if ra[e]["steps"] != rb[e]["steps"]:
            bad_l += 1
            len_pairs.append((ra[e]["steps"], rb[e]["steps"]))
            if bad_l <= 12:
                print("  LEN  t%d e%d  %d vs %d" % (t, e, ra[e]["steps"], rb[e]["steps"]))

print("\nepisodes compared   : %d" % n)
print("success-flag mismatches : %d" % bad_s)
print("episode-length mismatches: %d" % bad_l)
if len_pairs:
    import statistics
    d = [abs(x - y) for x, y in len_pairs]
    print("  length diffs: mean %.1f, max %d" % (statistics.mean(d), max(d)))

# per-task success and mean length, side by side
print("\ntask  succ_a succ_b   meanlen_a meanlen_b")
for t in range(10):
    a, b = load("n1_r0", t), load("n1_r1_r1", t)
    la = [r["steps"] for r in a["rows"]]
    lb = [r["steps"] for r in b["rows"]]
    print("%3d   %5d  %5d    %8.1f %8.1f"
          % (t, a["success"], b["success"], sum(la) / len(la), sum(lb) / len(lb)))