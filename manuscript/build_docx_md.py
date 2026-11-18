"""Flatten main.tex into a pandoc-friendly file and export Markdown and Word versions.

Cross-references are resolved from build/main.aux (run `tectonic --keep-intermediates --outdir build main.tex` first);
the TikZ figure and Algorithm 1 are replaced by PNG renderings in figures/.
Usage: python3 gen_aux.py && python3 build_docx_md.py path/to/elsevier.csl
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


def algorithm_text(m):
    """algorithmic block -> numbered lines (editable text in Word, no image)."""
    src = m.group(0)
    cap = re.search(r"\\caption\{(.*?)\}\\label", src).group(1)
    lines, depth, n = [], 0, 0
    for raw in src.splitlines():
        l = raw.strip()
        if not l or l.startswith(("\\begin", "\\end{algorithm", "\\caption", "\\small")):
            continue
        if l.startswith(("\\EndIf", "\\EndFor", "\\Else")):
            depth -= 1
            if l.startswith("\\Else"):
                n += 1; lines.append((n, depth, "\\textbf{else}")); depth += 1
            continue
        if l.startswith("\\Require"):
            lines.append((None, 0, "\\textbf{Require:} " + l[len("\\Require"):].strip())); continue
        l = re.sub(r"\\Comment\{([^}]*)\}", r" (\1)", l)
        inline_end = l.endswith("\\EndIf")
        l = l.replace("\\EndIf", "").strip()
        if l.startswith("\\State"):
            txt = l[len("\\State"):].strip()
        elif l.startswith("\\If"):
            i, dep = len("\\If{"), 1
            while dep:
                dep += {"{": 1, "}": -1}.get(l[i], 0); i += 1
            cond, rest = l[len("\\If{"):i - 1], l[i:].strip()
            txt = "\\textbf{if} " + cond + " \\textbf{then}" + ((" " + rest) if rest else "")
        elif l.startswith("\\For"):
            txt = "\\textbf{for} " + re.match(r"\\For\{(.*)\}", l).group(1) + " \\textbf{do}"
        else:
            txt = l
        n += 1
        lines.append((n, depth, txt.replace("\\textsc{Calibrate}", "\\textrm{CALIBRATE}")))
        if (l.startswith("\\If") and not inline_end) or l.startswith("\\For"):
            depth += 1
    out = [f"\n\n\\noindent\\textbf{{Algorithm 1.}} {cap}\n"]
    for num, d, txt in lines:
        pre = "" if num is None else f"{num}: "
        out.append("\n\\noindent{}" + "~" * (4 * d) + pre + txt + "\n")
    return "\n".join(out) + "\n"


body = re.sub(r"\\begin\{algorithm\}\[t\].*?\\end\{algorithm\}", algorithm_text, body, flags=re.S)
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


TITLE = re.search(r"\\title\{(.*?)\}\n", (HERE / "main.tex").read_text()).group(1)
AFFIL = ("Department of Statistics, Big Data Analyst Program, Manisa Vocational School of Technical Sciences, "
         "Manisa Celal Bayar University, Manisa 45140, T\\\"urkiye")
OUT = HERE / "build" / "word"
OUT.mkdir(parents=True, exist_ok=True)
REF = str(HERE / "reference.docx")


def pandoc_docx(tex, out, number=True):
    flat = HERE / "build" / (out.stem + ".tex")
    flat.write_text(tex)
    cmd = ["pandoc", str(flat), "-f", "latex", "--citeproc", "--bibliography", str(HERE / "refs.bib"), "--csl", CSL,
           "--resource-path", str(HERE), "-t", "docx", "--reference-doc", REF, "-o", str(out)]
    if number:
        cmd.insert(2, "--number-sections")
    subprocess.run(cmd, check=True)


named = (r"""\documentclass{article}\usepackage{amsmath,amssymb,graphicx,booktabs}\title{""" + TITLE + r"""}
\author{Osman Can \c{C}etlenbik\\""" + AFFIL + r"""\\Corresponding author. E-mail: osman.cetlenbik@cbu.edu.tr\\ORCID: 0009-0000-6561-1943}
\date{}\begin{document}\maketitle
""" + body + "\n\\section*{References}\n\\end{document}\n")
anon_body = re.sub(r"\\url\{https://github\.com/[^}]*\}", "[repository link removed for double-anonymised review]", body)
anon_body = anon_body.replace("osmancancet", "[anonymised]")
anon = (r"""\documentclass{article}\usepackage{amsmath,amssymb,graphicx,booktabs}\title{""" + TITLE + r"""}\date{}
\begin{document}\maketitle
""" + anon_body + "\n\\section*{References}\n\\end{document}\n")
pandoc_docx(named, OUT / "SLIDE_manuscript.docx")
pandoc_docx(anon, OUT / "SLIDE_manuscript_anonymized.docx")
title_page = (r"""\documentclass{article}\begin{document}
\section*{Title page}
\textbf{Title:} """ + TITLE + r"""

\textbf{Author:} Osman Can \c{C}etlenbik

\textbf{Affiliation:} """ + AFFIL + r"""

\textbf{Corresponding author:} Osman Can \c{C}etlenbik, osman.cetlenbik@cbu.edu.tr, ORCID 0009-0000-6561-1943

\textbf{Data and code availability:} All data are public. Code, per-run results and the manuscript source: \url{https://github.com/osmancancet/slide-llm-phishing-drift} (MIT licence).
\end{document}
""")
pandoc_docx(title_page, OUT / "SLIDE_title_page.docx", number=False)
SUB = HERE.parent / "submission"
for md, name in (("highlights.md", "SLIDE_highlights.docx"), ("cover_letter.md", "SLIDE_cover_letter.docx"),
                 ("response_to_reviewers.md", "SLIDE_response_to_KBS_reviewers.docx")):
    src = (SUB / md).read_text()
    if md == "cover_letter.md":          # keep the address block on separate lines
        a, b = src.split("\n\n", 1)
        src = "  \n".join(a.splitlines()) + "\n\n" + b
    subprocess.run(["pandoc", "-f", "markdown", "-t", "docx", "--reference-doc", REF, "-o", str(OUT / name)],
                   input=src, text=True, check=True)


def style_docx(path, double=True, line_numbers=True):
    """Times New Roman 12 pt, double spacing and continuous line numbers for the manuscript files."""
    from docx import Document
    from docx.shared import Pt
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    d = Document(path)
    for st in d.styles:
        try:
            f = st.font
        except AttributeError:
            continue
        f.name = "Times New Roman"
        rpr = st.element.get_or_add_rPr()
        rf = rpr.find(qn("w:rFonts"))
        if rf is None:
            rf = OxmlElement("w:rFonts"); rpr.append(rf)
        for a in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
            rf.set(qn(a), "Times New Roman")
        for a in ("w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"):
            rf.attrib.pop(qn(a), None)
    d.styles["Normal"].font.size = Pt(12)
    for nm in ("Body Text", "First Paragraph", "Normal", "Compact"):
        if nm in [x.name for x in d.styles] and double:
            pf = d.styles[nm].paragraph_format
            pf.line_spacing = 2.0
    for t in d.tables:
        tbl = t._tbl
        borders = OxmlElement("w:tblBorders")
        for edge in ("top", "bottom", "insideH"):
            e = OxmlElement(f"w:{edge}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:color"), "000000")
            borders.append(e)
        tbl.tblPr.append(borders)
    if line_numbers:
        for sec in d.sections:
            if sec._sectPr.find(qn("w:lnNumType")) is not None:
                continue
            ln = OxmlElement("w:lnNumType"); ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
            sec._sectPr.append(ln)
    d.core_properties.author = "Osman Can Çetlenbik" if "anonym" not in path.name else ""
    d.core_properties.last_modified_by = ""
    d.save(path)


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



for f in ("SLIDE_manuscript.docx", "SLIDE_manuscript_anonymized.docx"):
    tidy_docx(OUT / f)
    style_docx(OUT / f, double=True, line_numbers=True)
for f in ("SLIDE_title_page.docx", "SLIDE_highlights.docx", "SLIDE_cover_letter.docx", "SLIDE_response_to_KBS_reviewers.docx"):
    style_docx(OUT / f, double=False, line_numbers=False)
print("wrote", sorted(x.name for x in OUT.glob("*.docx")))
