"""Flatten main.tex into a pandoc-friendly file and export Markdown and Word versions.

Cross-references are resolved from build/main.aux (run `tectonic --keep-intermediates --outdir build main.tex` first);
the TikZ figure and Algorithm 1 are replaced by PNG renderings in figures/.
Usage: python3 build_docx_md.py path/to/ieee.csl
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSL = sys.argv[1]
aux = (HERE / "build" / "main.aux").read_text()
LAB = {m.group(1): m.group(2) for m in re.finditer(r"\\newlabel\{([^}]*)\}\{\{([^}]*)\}", aux)}


def inline(text):
    return re.sub(r"\\input\{([^}]*)\}", lambda m: inline((HERE / (m.group(1) if m.group(1).endswith(".tex") else m.group(1) + ".tex")).read_text()), text)


body = inline((HERE / "main.tex").read_text())
body = body[body.index("\\begin{abstract}"):body.index("\\bibliographystyle")]
body = body.replace("\\end{frontmatter}", "")
# keywords -> plain paragraph
body = re.sub(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", lambda m: "\\noindent\\textbf{Keywords:} " + re.sub(r"\s*\\sep\s*", "; ", m.group(1)).strip() + "\n", body, flags=re.S)
# TikZ figure and algorithm -> images
body = re.sub(r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}", r"\\includegraphics[width=\\linewidth]{figures/arch.png}", body, flags=re.S)
body = re.sub(r"\\begin\{algorithm\}\[t\].*?\\end\{algorithm\}",
              r"\\begin{figure}[t]\n\\includegraphics[width=\\linewidth]{figures/algorithm1.png}\n\\caption{SLIDE prequential detection with label-on-alarm adaptation.}\n\\end{figure}", body, flags=re.S)
body = re.sub(r"figures/(\w+)\.pdf", r"figures/\1.png", body)
# table layout helpers pandoc does not parse
body = body.replace("\\resizebox{\\textwidth}{!}{\\begin{tabular}", "\\begin{tabular}").replace("\\end{tabular}}", "\\end{tabular}")
body = body.replace(">{\\raggedright\\arraybackslash}", "")
body = re.sub(r"\\setlength\{\\tabcolsep\}\{[^}]*\}", "", body)
body = re.sub(r"\\cmidrule(\([^)]*\))?\{[^}]*\}", "", body)
body = re.sub(r"\\(scriptsize|small|tiny)\b", "", body)
body = re.sub(r"\{\s*\\url\{([^}]*)\}\}", r" doi:\1", body)
body = body.replace("\\newline", " ")
# labelled list items keep their labels
body = re.sub(r"\\item\[([^]]*)\]", r"\\item \\textbf{\1} ", body)
body = body.replace("\\cmark", "\u2713").replace("\\xmark", "\u2717").replace("\\pmark", "\u25cb")


# numbered captions for floats, emitted as ordinary paragraphs so that every output format keeps them
def take_caption(inner):
    i = inner.find("\\caption{")
    if i < 0:
        return inner, ""
    j, depth = i + len("\\caption{"), 1
    while depth:
        depth += {"{": 1, "}": -1}.get(inner[j], 0)
        j += 1
    return inner[:i] + inner[j:], inner[i + len("\\caption{"):j - 1]


def number_float(m):
    env, inner = m.group(1), m.group(2)
    lab = re.search(r"\\label\{([^}]*)\}", inner)
    inner, cap = take_caption(inner)
    if "algorithm1.png" in inner:
        kind, num = "Algorithm", "1"
    else:
        kind = "Table" if env.startswith("table") else "Figure"
        num = LAB.get(lab.group(1), "?") if lab else "?"
    para = f"\n\n\\noindent\\textbf{{{kind} {num}.}} {cap}\n\n"
    inner = re.sub(r"\\label\{[^}]*\}", "", inner).replace("\\centering", "")
    return para + inner if kind == "Table" else inner + para


body = re.sub(r"\\begin\{(table\*?|figure\*?)\}(?:\[[^]]*\])?(.*?)\\end\{\1\}", number_float, body, flags=re.S)
# display equations: Word drops \\tag, so the number is written into the equation, in the PDF's order
eqn = iter(range(1, 1000))
body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}",
              lambda m: "\\[" + re.sub(r"\\label\{[^}]*\}", "", m.group(1)).strip() + f"\\qquad({next(eqn)})\\]", body, flags=re.S)
# paragraph headings become bold run-in labels instead of numbered level-4 headings
body = re.sub(r"\\paragraph\{([^}]*)\}", r"\n\n\\noindent\\textbf{\1.} ", body)
body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({LAB.get(m.group(1), '?')})", body)
body = re.sub(r"\\ref\{([^}]*)\}", lambda m: LAB.get(m.group(1), "?"), body)
body = re.sub(r"\\label\{[^}]*\}", "", body)
body = re.sub(r"\\begin\{proposition\}\[([^]]*)\]", lambda m: "\n\n\\noindent\\textbf{Proposition. (" + m.group(1) + ").} ", body)
body = body.replace("\\end{proposition}", "")
body = body.replace("\\begin{proof}", "\\emph{Proof.} ").replace("\\end{proof}", " $\\square$\n")
body = re.sub(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", r"\\section*{Abstract}\1", body, flags=re.S)
# number propositions in order of appearance
k = iter(range(1, 100))
body = re.sub(r"\\textbf\{Proposition\. ", lambda m: f"\\textbf{{Proposition {next(k)} ", body)

pre = r"""\documentclass{article}
\usepackage{amsmath,amssymb,graphicx,booktabs}
\title{SLIDE: label-free sliced-Wasserstein drift detection and budgeted adaptation for LLM-generated phishing in e-mail streams}
\author{Osman Can \c{C}etlenbik\\Big Data Analytics Program, Department of Statistics, Manisa Technical Sciences Vocational School, Manisa Celal Bayar University, Manisa 45140, T\"urkiye\\Corresponding author. E-mail: osman.cetlenbik@cbu.edu.tr}
\begin{document}
\maketitle
"""
flat = HERE / "build" / "flat_pandoc.tex"
flat.write_text(pre + body + "\n\\section*{References}\n\\end{document}\n")
common = ["pandoc", str(flat), "-f", "latex", "--citeproc", "--bibliography", str(HERE / "refs.bib"), "--csl", CSL,
          "--number-sections", "--resource-path", str(HERE)]
subprocess.run(common + ["-t", "docx", "--reference-doc", str(HERE / "reference.docx"), "-o", str(HERE / "main.docx")], check=True)


def tidy_docx(path):
    """Clean document properties and give table cells a compact 9 pt style."""
    import zipfile
    src = zipfile.ZipFile(path)
    parts = {n: src.read(n) for n in src.namelist()}
    src.close()
    doc = parts["word/document.xml"].decode()
    doc = re.sub(r"<w:tbl>.*?</w:tbl>", lambda m: m.group(0).replace('w:val="Compact"', 'w:val="TableText"'), doc, flags=re.S)
    # wide tables go on landscape pages: a section break before the caption and after the table
    ftr = re.findall(r'<w:footerReference[^>]*r:id="(\w+)"', doc)
    ftr = f'<w:footerReference w:type="default" r:id="{ftr[-1]}"/>' if ftr else ""
    sect = ('<w:sectPr>' + ftr + '{sz}'
            '<w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440" w:header="720" w:footer="720" w:gutter="0"/>'
            '<w:lnNumType w:countBy="1" w:restart="continuous"/></w:sectPr>')
    portrait = sect.format(sz='<w:pgSz w:w="11906" w:h="16838"/>')
    landscape = sect.format(sz='<w:pgSz w:w="16838" w:h="11906" w:orient="landscape"/>')
    tables = list(re.finditer(r"<w:tbl>.*?</w:tbl>", doc, flags=re.S))
    for m in reversed(tables):
        ncol = m.group(0).count("<w:gridCol")
        before = doc[:m.start()]
        cap = max(before.rfind("<w:p>"), before.rfind("<w:p "))
        while cap > 0 and "Table " not in re.sub(r"<[^>]+>", "", doc[cap:m.start()]):
            cap = max(doc.rfind("<w:p>", 0, cap), doc.rfind("<w:p ", 0, cap))
        captext = re.sub(r"<[^>]+>", "", doc[cap:m.start()])
        if ncol >= 9 or captext.strip().startswith("Table 1."):
            doc = (doc[:cap] + f"<w:p><w:pPr>{portrait}</w:pPr></w:p>" + doc[cap:m.end()]
                   + f"<w:p><w:pPr>{landscape}</w:pPr></w:p>" + doc[m.end():])
    # full-width tables whose columns Word sizes to their content
    doc = re.sub(r"<w:tblW [^>]*/>", '<w:tblW w:type="pct" w:w="5000"/><w:tblLayout w:type="autofit"/>', doc)
    doc = re.sub(r"<w:tblGrid>.*?</w:tblGrid>", "", doc, flags=re.S)
    doc = re.sub(r"<w:tcW [^>]*/>", '<w:tcW w:type="auto" w:w="0"/>', doc)
    parts["word/document.xml"] = doc.encode()
    sty = parts["word/styles.xml"].decode()
    if 'w:styleId="TableText"' not in sty:
        sty = sty.replace("</w:styles>", '<w:style w:type="paragraph" w:customStyle="1" w:styleId="TableText"><w:name w:val="Table Text"/>'
                          '<w:basedOn w:val="Normal"/><w:pPr><w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/></w:pPr>'
                          '<w:rPr><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style></w:styles>')
    parts["word/styles.xml"] = sty.encode()
    core = parts["docProps/core.xml"].decode()
    core = re.sub(r"<dc:creator>.*?</dc:creator>", "<dc:creator>Osman Can Çetlenbik</dc:creator>", core, flags=re.S)
    parts["docProps/core.xml"] = core.encode()
    parts.pop("docProps/custom.xml", None)
    ct = parts["[Content_Types].xml"].decode().replace('<Override PartName="/docProps/custom.xml" ContentType="application/vnd.openxmlformats-officedocument.custom-properties+xml" />', "")
    ct = re.sub(r'<Override PartName="/docProps/custom.xml"[^>]*/>', "", ct)
    parts["[Content_Types].xml"] = ct.encode()
    rels = parts["_rels/.rels"].decode()
    rels = re.sub(r'<Relationship [^>]*Target="docProps/custom.xml"[^>]*/>', "", rels)
    parts["_rels/.rels"] = rels.encode()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as out:
        for n, b in parts.items():
            out.writestr(n, b)


tidy_docx(HERE / "main.docx")
md = subprocess.run(common + ["-t", "gfm-smart", "--wrap=none"], check=True, capture_output=True, text=True).stdout
head = ("# SLIDE: label-free sliced-Wasserstein drift detection and budgeted adaptation for LLM-generated phishing in e-mail streams\n\n"
        "**Osman Can Çetlenbik**  \nBig Data Analytics Program, Department of Statistics, Manisa Technical Sciences Vocational School, Manisa Celal Bayar University, "
        "Manisa 45140, Türkiye  \nCorresponding author. E-mail: osman.cetlenbik@cbu.edu.tr\n\n")
md = re.sub(r'<span class="nocase">(.*?)</span>', r"\1", md)
(HERE / "main.md").write_text(head + md)
print("wrote", HERE / "main.docx", HERE / "main.md")
