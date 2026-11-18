"""Write build/main.aux-style \\newlabel lines without a TeX engine (elsarticle numbering:
sections 1, 1.1; tables and table* share one counter; figures likewise; equations, propositions, algorithms)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def inline(text):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: inline((HERE / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")).read_text()), text)


body = inline((HERE / "main.tex").read_text())
body = re.sub(r"(?<!\\)%.*", "", body)
tok = re.compile(r"\\(section|subsection|subsubsection)\*?\{|\\begin\{(table\*?|figure\*?|equation|proposition|algorithm)\}|\\end\{(table\*?|figure\*?|algorithm)\}|\\label\{([^}]*)\}")
c = dict(sec=0, sub=0, subsub=0, table=0, figure=0, equation=0, proposition=0, algorithm=0)
cur, flt, out = "", None, {}
for m in tok.finditer(body):
    if m.group(1):
        star = body[m.start():m.end()].endswith("*{")
        if star:
            continue
        if m.group(1) == "section":
            c["sec"] += 1; c["sub"] = 0; cur = str(c["sec"])
        elif m.group(1) == "subsection":
            c["sub"] += 1; c["subsub"] = 0; cur = f"{c['sec']}.{c['sub']}"
        else:
            c["subsub"] += 1; cur = f"{c['sec']}.{c['sub']}.{c['subsub']}"
    elif m.group(2):
        k = m.group(2).rstrip("*")
        c[k] += 1
        if k in ("table", "figure", "algorithm"):
            flt = str(c[k])
        else:
            cur_eq = str(c[k]); flt_eq = cur_eq
            flt = None
            out["__last"] = cur_eq
    elif m.group(3):
        flt = None
    elif m.group(4):
        lab = m.group(4)
        if lab.startswith(("eq:", "prop:")):
            out[lab] = out["__last"]
        elif flt is not None:
            out[lab] = flt
        else:
            out[lab] = cur
out.pop("__last", None)
(HERE / "build").mkdir(exist_ok=True)
(HERE / "build" / "main.aux").write_text("".join(f"\\newlabel{{{k}}}{{{{{v}}}{{}}}}\n" for k, v in out.items()))
if __name__ == "__main__":
    for k, v in out.items():
        print(k, v)
