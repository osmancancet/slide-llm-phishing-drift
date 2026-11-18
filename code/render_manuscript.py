"""Fill numeric tokens in manuscript/tpl/*.tex from results/revision/numbers.json -> manuscript/sections/.

Token syntax: {{key}} or {{key:fmt}} where key is a numbers.json key and fmt a Python format spec
(e.g. .3f, .0f, .1%). {{key:,.0f}} prints LaTeX thousands separators. Unknown keys abort the build.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
N = json.load(open(ROOT / "results" / "revision" / "numbers.json"))
TOK = re.compile(r"\{\{([^{}:]+?)(?::([^{}]+))?\}\}")
missing = []


LAB = {}


def rep(m):
    k, f = m.group(1), m.group(2) or ".3f"
    if k == "ref":                              # {{ref:label}}: section/table number from the generated aux
        if f not in LAB:
            missing.append("ref:" + f)
            return "??"
        return LAB[f]
    if k.startswith("="):                       # {{=[key1]-[key2]:fmt}}: arithmetic on numbers.json values
        keys = re.findall(r"\[([^\[\]]+)\]", k)
        bad = [x for x in keys if N.get(x) is None]
        if bad:
            missing.extend(bad)
            return "??"
        v = eval(re.sub(r"\[([^\[\]]+)\]", lambda q: repr(float(N[q.group(1)])), k[1:]), {"abs": abs})
        return fmt(v, f)
    if k not in N or N[k] is None:
        missing.append(k)
        return "??"
    return fmt(N[k], f)


def fmt(v, f):
    if f.startswith(","):
        return format(v, f).replace(",", "{,}")
    if f.endswith("%"):
        return format(v, f).replace("%", "\\%")
    out = format(v, f)
    return out.replace("-", "$-$", 1) if out.startswith("-") else out


for p in sorted((ROOT / "manuscript" / "tpl").glob("*.tex")):
    out = TOK.sub(rep, p.read_text())
    (ROOT / "manuscript" / "sections" / p.name).write_text(out)
# cross-reference numbers for the submission texts
import subprocess
subprocess.run([sys.executable, str(ROOT / "manuscript" / "gen_aux.py")], check=True, capture_output=True)
LAB.update(re.findall(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", (ROOT / "manuscript" / "build" / "main.aux").read_text()))
# submission texts (response letter, cover letter, highlights): submission/tpl/*.md -> submission/*.md
(ROOT / "submission").mkdir(exist_ok=True)
for p in sorted((ROOT / "submission" / "tpl").glob("*.md")):
    (ROOT / "submission" / p.name).write_text(TOK.sub(rep, p.read_text()).replace("$-$", "-").replace("{,}", ",").replace("\\%", "%"))
if missing:
    print("MISSING:", sorted(set(missing)))
    sys.exit(1)
print("rendered")
