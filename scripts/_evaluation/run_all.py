#!/usr/bin/env python
"""
run_all.py — bedieningspaneel voor de evaluaties
================================================
Eén ingang om resultaten, tabellen en figuren opnieuw te genereren in een nette,
gedateerde map:  outputs/runs/<datum_tijd>/{results,aggregate,figures,logs}.

Start met je eval-env, bijv.:
    python scripts/_evaluation/run_all.py

De hele test opnieuw draaien (V-JEPA 2 als default V-JEPA, dus zonder V-JEPA 1):
    python scripts/_evaluation/run_all.py --preset default --post all --yes

V-JEPA 1 vs 2 vergelijking erbij (V-JEPA 1 is opt-in):
    python scripts/_evaluation/run_all.py --preset all --post all --yes
    # of alleen V-JEPA 1:  --preset vjepa

Sneltoetsen voor gevorderden (zonder menu):
    --select loso_videomae,cda_dinov2   --post aggregate,figures   --yes
    --preset fast   --post all   --yes
    --post figures,carepd --run latest --yes      (alleen verwerken van laatste run)
    --list
"""

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

EVAL_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parents[1]
OUTPUTS      = PROJECT_ROOT / "outputs"
RUNS_DIR     = OUTPUTS / "runs"
FEATURE_DIR  = PROJECT_ROOT / "assets" / "datasets" / "fabricated_datasets"

# ── kleur (gaat uit bij NO_COLOR of niet-terminal) ────────────────────────────
_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
def c(s, *codes):
    return ("".join(f"\033[{k}m" for k in codes) + s + "\033[0m") if _COLOR else s
B, D, GRN, YEL, RED, CYN = 1, 2, 32, 33, 31, 36

# ── register ──────────────────────────────────────────────────────────────────
FEAT = {"videomae": "videomae_features_multilayer.pkl",
        "vjepa": "vjepa_features.pkl", "vjepa2": "vjepa2_features.pkl",
        "dinov2": "dinov2_features_81f.pkl"}
# (key, protocol, model, script, speed, feature, ~minuten)
EVALS = [
    ("loso_videomae",      "loso", "VideoMAE",      "loso/videomae_loso.py",      "fast", FEAT["videomae"], 10),
    ("loso_vjepa",         "loso", "V-JEPA",        "loso/vjepa_loso.py",         "fast", FEAT["vjepa"],     3),
    ("loso_vjepa2",        "loso", "V-JEPA 2",      "loso/vjepa2_loso.py",        "fast", FEAT["vjepa2"],    3),
    ("loso_dinov2",        "loso", "DINOv2-BiLSTM", "loso/dinov2_loso.py",        "slow", FEAT["dinov2"],  180),
    ("mida_videomae",      "mida", "VideoMAE",      "mida/videomae_mida.py",      "fast", FEAT["videomae"],  6),
    ("mida_vjepa",         "mida", "V-JEPA",        "mida/vjepa_mida.py",         "fast", FEAT["vjepa"],     3),
    ("mida_vjepa2",        "mida", "V-JEPA 2",      "mida/vjepa2_mida.py",        "fast", FEAT["vjepa2"],    3),
    ("mida_dinov2",        "mida", "DINOv2",        "mida/dinov2_mida.py",        "fast", FEAT["dinov2"],    4),
    ("mida_dinov2_bilstm", "mida", "DINOv2-BiLSTM", "mida/dinov2_mida_bilstm.py", "slow", FEAT["dinov2"],   30),
    ("lodo_videomae",      "lodo", "VideoMAE",      "lodo/videomae_lodo.py",      "fast", FEAT["videomae"],  3),
    ("lodo_vjepa",         "lodo", "V-JEPA",        "lodo/vjepa_lodo.py",         "fast", FEAT["vjepa"],     2),
    ("lodo_vjepa2",        "lodo", "V-JEPA 2",      "lodo/vjepa2_lodo.py",        "fast", FEAT["vjepa2"],    2),
    ("lodo_dinov2",        "lodo", "DINOv2",        "lodo/dinov2_lodo.py",        "fast", FEAT["dinov2"],    2),
    ("lodo_dinov2_bilstm", "lodo", "DINOv2-BiLSTM", "lodo/dinov2_lodo_bilstm.py", "slow", FEAT["dinov2"],   20),
    ("cda_videomae",       "cda",  "VideoMAE",      "cda/videomae_cda.py",        "fast", FEAT["videomae"],  3),
    ("cda_vjepa",          "cda",  "V-JEPA",        "cda/vjepa_cda.py",           "fast", FEAT["vjepa"],     1),
    ("cda_vjepa2",         "cda",  "V-JEPA 2",      "cda/vjepa2_cda.py",          "fast", FEAT["vjepa2"],    1),
    ("cda_dinov2",         "cda",  "DINOv2",        "cda/dinov2_cda.py",          "fast", FEAT["dinov2"],    2),
    ("cda_dinov2_bilstm",  "cda",  "DINOv2-BiLSTM", "cda/dinov2_cda_bilstm.py",   "slow", FEAT["dinov2"],   25),
]
POST = [("aggregate", "aggregate_results.py", "tabellen", 1),
        ("figures",   "plot_results.py",      "grafieken", 2),
        ("carepd",    "baseline_comparison.py", "Care-PD-vergelijking", 1),
        ("confusion", "confusion_tables.py",  "confusion-tabellen", 1)]
EVAL_BY_KEY = {e[0]: e for e in EVALS}
POST_BY_KEY = {p[0]: p for p in POST}
POST_KEYS   = [p[0] for p in POST]


def label(key):
    _, proto, model, *_ = EVAL_BY_KEY[key]
    return f"{proto.upper()}·{model}"

def htime(m):
    return f"~{round(m/60)}u" if m >= 90 else f"~{m}m"


# ── paneel ────────────────────────────────────────────────────────────────────

def print_menu():
    print()
    print(c("  BEP · evaluaties", B))
    print(c("  LOSO binnen 1 dataset   CDA dataset→dataset   "
            "LODO alles−1→die ene   MIDA LOSO+extra data", D))
    print()
    for i, (key, proto, model, _s, speed, _f, mins) in enumerate(EVALS, 1):
        mark = c(" ▲", YEL) if speed == "slow" else "  "
        print(f"  {c(f'{i:>2}', B)}  {proto.upper():4s} · {model:14s}{mark} {c(htime(mins), D)}")
    parts = [f"{c(str(len(EVALS) + 1 + j), B)} {p[2]}" for j, p in enumerate(POST)]
    print(c("  ── na afloop ──", D))
    print("  " + "    ".join(parts))
    print(c("  ▲ traag (BiLSTM)   presets: default(=alles, V-JEPA 2) all fast slow "
            "loso cda lodo mida   dinov2 videomae vjepa(=V-JEPA 1) vjepa2", D))


# ── selectie ──────────────────────────────────────────────────────────────────

def resolve_tokens(tokens):
    eval_keys, post_keys = [], []
    n = len(EVALS)
    def add_e(k):
        if k not in eval_keys: eval_keys.append(k)
    def add_p(k):
        if k not in post_keys: post_keys.append(k)

    for raw in tokens:
        t = raw.strip().lower()
        if not t:
            continue
        if "-" in t and all(p.isdigit() for p in t.split("-", 1)):
            a, b = (int(x) for x in t.split("-", 1))
            for i in range(a, b + 1):
                if 1 <= i <= n: add_e(EVALS[i - 1][0])
                elif n < i <= n + len(POST): add_p(POST[i - 1 - n][0])
        elif t.isdigit():
            i = int(t)
            if 1 <= i <= n: add_e(EVALS[i - 1][0])
            elif n < i <= n + len(POST): add_p(POST[i - 1 - n][0])
        elif t == "all":
            for e in EVALS: add_e(e[0])
            for p in POST:  add_p(p[0])
        elif t in ("default", "main"):
            # the headline test: every protocol/model EXCEPT V-JEPA 1.
            # V-JEPA 2 is the default V-JEPA; V-JEPA 1 stays opt-in (preset
            # `vjepa`) so you can reproduce the V-JEPA 1 vs 2 comparison.
            for e in EVALS:
                if e[3].split("/")[-1].split("_")[0] != "vjepa": add_e(e[0])
            for p in POST: add_p(p[0])
        elif t in ("fast", "slow"):
            for e in EVALS:
                if e[4] == t: add_e(e[0])
        elif t in ("loso", "mida", "lodo", "cda"):
            for e in EVALS:
                if e[1] == t: add_e(e[0])
        elif t in ("videomae", "vjepa", "vjepa2", "dinov2"):
            # match on the script filename's model token (vjepa vs vjepa2 disambiguated)
            for e in EVALS:
                if e[3].split("/")[-1].split("_")[0] == t: add_e(e[0])
        elif t == "post":
            for p in POST: add_p(p[0])
        elif t in POST_KEYS: add_p(t)
        elif t in EVAL_BY_KEY: add_e(t)
        else:
            print(c(f"  ! genegeerd: {raw}", YEL))
    eval_keys = [e[0] for e in EVALS if e[0] in eval_keys]
    post_keys = [k for k in POST_KEYS if k in post_keys]
    return eval_keys, post_keys


# ── run-map helpers ───────────────────────────────────────────────────────────

def new_run_dir():
    d = RUNS_DIR / dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=False)
    return d

def list_existing_runs():
    return sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()),
                  key=lambda p: p.name) if RUNS_DIR.exists() else []

def resolve_run_arg(run_arg):
    runs = list_existing_runs()
    if not runs:
        return None
    if run_arg in (None, "latest"):
        return runs[-1]
    cand = RUNS_DIR / run_arg
    return cand if cand.is_dir() else None

def update_latest_symlink(run_dir):
    link = OUTPUTS / "latest"
    try:
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(Path("runs") / run_dir.name)
    except OSError as e:
        print(c(f"  ! 'latest' symlink niet bijgewerkt: {e}", YEL))

def git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=PROJECT_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# ── uitvoeren ─────────────────────────────────────────────────────────────────

def run_step(key, script_rel, run_dir, log_dir, name):
    log_path = log_dir / (key + ".log")
    env = {**os.environ, "BEP_OUTPUT_ROOT": str(run_dir)}
    print(c(f"\n  ▶ {name}", B) + c(f"  ({script_rel})", D))
    start = dt.datetime.now()
    with open(log_path, "w") as logf:
        proc = subprocess.Popen([sys.executable, str(EVAL_DIR / script_rel)],
                                cwd=str(PROJECT_ROOT), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write("    " + line)
            logf.write(line)
        proc.wait()
    dur = (dt.datetime.now() - start).total_seconds()
    ok = proc.returncode == 0
    print(("    " + c("✓", GRN) if ok else "    " + c("✗", RED)) + f" {name} ({dur:.0f}s)")
    return {"label": name, "key": key, "script": script_rel,
            "ok": ok, "returncode": proc.returncode, "seconds": round(dur, 1),
            "log": str(log_path.relative_to(PROJECT_ROOT))}

def preflight(eval_keys):
    ok = []
    for k in eval_keys:
        if (FEATURE_DIR / EVAL_BY_KEY[k][5]).exists():
            ok.append(k)
        else:
            print(c(f"  ! {label(k)} overgeslagen — features ontbreken "
                    f"({EVAL_BY_KEY[k][5]})", YEL))
    return ok


def execute(eval_keys, post_keys, assume_yes, run_arg):
    eval_keys = preflight(eval_keys)
    if not eval_keys and not post_keys:
        print("  Niets te doen."); return 0

    if eval_keys:
        run_dir, run_mode = None, "nieuw"
    else:
        run_dir = resolve_run_arg(run_arg)
        if run_dir is None:
            print(c("  Geen bestaande run om te verwerken.", YEL)); return 1
        run_mode = run_dir.name

    have_agg = run_dir is not None and (run_dir / "aggregate").exists()
    if "carepd" in post_keys and "aggregate" not in post_keys and not have_agg:
        post_keys = ["aggregate"] + post_keys

    total = sum(EVAL_BY_KEY[k][6] for k in eval_keys) + sum(POST_BY_KEY[k][3] for k in post_keys)
    slow = any(EVAL_BY_KEY[k][4] == "slow" for k in eval_keys)

    # ── compact bevestigingsblok ──
    print()
    print(c(f"  → outputs/runs/{run_mode}", B))
    if eval_keys:
        print("  doen : " + ", ".join(label(k) for k in eval_keys))
    if post_keys:
        print("  daarna: " + ", ".join(POST_BY_KEY[k][2] for k in post_keys))
    print(c(f"  tijd : {htime(total)}" + ("  ▲ incl. trage stappen" if slow else ""), D))
    if not assume_yes:
        try:
            if input(c("  start? [j/N] ", B)).strip().lower() not in ("j", "ja", "y", "yes"):
                print("  Geannuleerd."); return 0
        except EOFError:
            print("  (geen invoer) Geannuleerd."); return 0

    if run_dir is None:
        run_dir = new_run_dir()
    log_dir = run_dir / "logs"; log_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for k in eval_keys:
        results.append(run_step(k, EVAL_BY_KEY[k][3], run_dir, log_dir, label(k)))
    for k in post_keys:
        results.append(run_step(k, POST_BY_KEY[k][1], run_dir, log_dir, POST_BY_KEY[k][2]))

    update_latest_symlink(run_dir)
    with open(run_dir / "run_manifest.json", "w") as f:
        json.dump({"timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                   "run_dir": str(run_dir.relative_to(PROJECT_ROOT)),
                   "git_sha": git_sha(), "python": sys.executable,
                   "evaluations": eval_keys, "post_steps": post_keys,
                   "steps": results}, f, indent=2)

    # ── compacte samenvatting ──
    n_fail = sum(1 for r in results if not r["ok"])
    print()
    print(c(f"  klaar → outputs/runs/{run_dir.name}", B) + c("  (= outputs/latest)", D))
    for r in results:
        tick = c("✓", GRN) if r["ok"] else c("✗", RED)
        print(f"  {tick} {r['label']:22s} " + c(f"{r['seconds']:.0f}s", D))
    print((c(f"  {len(results) - n_fail}/{len(results)} ok", GRN) if not n_fail
           else c(f"  {len(results) - n_fail}/{len(results)} ok — {n_fail} mislukt (zie logs)", RED)))
    return 1 if n_fail else 0


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(add_help=True, description="BEP evaluatie-bedieningspaneel")
    ap.add_argument("--select"); ap.add_argument("--preset"); ap.add_argument("--post")
    ap.add_argument("--run"); ap.add_argument("--list", action="store_true")
    ap.add_argument("--yes", "-y", action="store_true")
    args = ap.parse_args()

    if args.list:
        print_menu(); return

    tokens = []
    if args.preset: tokens.append(args.preset)
    if args.select: tokens += args.select.replace(",", " ").split()
    if args.post:   tokens += (POST_KEYS if args.post.strip().lower() == "all"
                               else args.post.replace(",", " ").split())

    if not tokens:                                  # interactief paneel
        print_menu()
        try:
            tokens = input(c("\n  kies (bv. 1,4  of  fast): ", B)).replace(",", " ").split()
        except EOFError:
            print("  (geen invoer)"); return
        if not tokens:
            print("  Niets gekozen."); return

    eval_keys, post_keys = resolve_tokens(tokens)
    sys.exit(execute(eval_keys, post_keys, assume_yes=args.yes, run_arg=args.run))


if __name__ == "__main__":
    main()
