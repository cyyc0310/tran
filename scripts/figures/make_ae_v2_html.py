#!/usr/bin/env python
"""Typeset the paper draft (v2) into a self-contained academic HTML.

Input : docs/paper/2026-09-18-ae-submission-zh-v2.md (verbatim source; journal meta removed)
Output: docs/paper/transcif_v2_zh.html (styles + KaTeX + images inlined)

Rendering chain
  md -> pandoc -> HTML body (native tables)
  KaTeX auto-render for $$...$$ / \\(...\\) / $...$
  Images: figures/*.png referenced by the md -> base64 data URIs
  Tables: three-line (booktabs) styling
A4 print layout: tables/images never split; print to PDF via browser.
"""
from pathlib import Path
import base64
import re
import subprocess

ROOT = Path("/Users/cyyc0310/code/tran")
PAPER = ROOT / "docs/paper/2026-09-18-ae-submission-zh-v2.md"
OUT = ROOT / "docs/paper/transcif_ae_v2_zh.html"
FIG = ROOT / "figures"

body = subprocess.run(
    ["pandoc", PAPER.name, "-f",
     "markdown+tex_math_dollars+pipe_tables+tex_math_single_backslash",
     "-t", "html"],
    cwd=str(ROOT / "docs/paper"),
    capture_output=True, text=True, check=True).stdout


def inline_img(m):
    src = m.group(1)
    name = src.split("/")[-1]
    p = FIG / name
    if not p.exists():
        return m.group(0)
    b64 = base64.b64encode(p.read_bytes()).decode()
    return 'src="data:image/png;base64,' + b64 + '"'


body = re.sub(r'src="([^"]+\.png)"', inline_img, body)

KATEX = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function() {
  renderMathInElement(document.body, {
    delimiters: [
      {left: "$$", right: "$$", display: true},
      {left: "\\\\(\\\\)", right: "\\\\)\\\\)", display: false},
      {left: "$", right: "$", display: false}
    ],
    throwOnError: false
  });
});
</script>"""

CSS = """
:root {
  --ink: #1a1a1a;
  --rule: #2c3e50;
  --soft: #b8b8b8;
  --accent: #2471a3;
}
* { box-sizing: border-box; }
body {
  font-family: "Songti SC", "Noto Serif CJK SC", "Source Han Serif SC", "SimSun", serif;
  font-size: 10.5pt; line-height: 1.95; color: var(--ink);
  max-width: 170mm; margin: 0 auto; padding: 20mm 0 28mm;
  background: #fff; text-align: justify;
}
h1 { font-size: 17pt; font-weight: 700; text-align: center;
     margin: 2mm 0 9mm; line-height: 1.6; }
h2 { font-size: 13pt; font-weight: 700; border-bottom: 1.5pt solid var(--rule);
     padding-bottom: 2mm; margin: 15mm 0 6mm; break-after: avoid; }
h3 { font-size: 11.5pt; font-weight: 700; margin: 9mm 0 4mm; break-after: avoid; }
h4 { font-size: 10.5pt; font-weight: 700; margin: 6.5mm 0 3mm; break-after: avoid; }
p { margin: 0 0 3.4mm; }
li { margin: 1.4mm 0; }
ul, ol { margin: 1.5mm 0 3.5mm; padding-left: 7mm; }
table { border-collapse: collapse; margin: 7mm auto 8mm; font-size: 9pt;
        width: auto; max-width: 100%; line-height: 1.55; }
th, td { padding: 2mm 3.2mm; text-align: center; vertical-align: middle; }
thead tr { border-top: 1.2pt solid var(--rule);
           border-bottom: 0.7pt solid var(--rule); }
tbody tr:last-child { border-bottom: 1.2pt solid var(--rule); }
tbody tr:not(:last-child) { border-bottom: 0.35pt solid #d9d9d9; }
th { font-weight: 700; }
td:first-child, th:first-child { text-align: left; }
figure { margin: 6mm 0; }
img { max-width: 92%; height: auto; display: block; margin: 6mm auto 2.5mm; }
p:has(> strong:first-child) { text-align: center; font-size: 9.5pt;
                              color: #333; margin: 1.5mm 0 4.5mm; line-height: 1.6; }
p:has(> img) { break-inside: avoid; page-break-inside: avoid; }
p:has(> img) + p:has(> strong:first-child) { break-before: avoid; }
hr { border: none; border-top: 0.6pt solid var(--soft); margin: 10mm 0; }
blockquote { margin: 5mm 0; padding: 3.5mm 7mm;
             border-left: 2.5pt solid var(--accent);
             background: #f6f8fa; font-size: 10pt; }
code { font-family: "SF Mono", Menlo, Consolas, monospace; font-size: 9pt;
       background: #f4f4f4; padding: 0.4mm 1.4mm; border-radius: 2px; }
.katex-display { margin: 5.5mm 0; overflow-x: auto; overflow-y: hidden; }
@page { size: A4; margin: 24mm 22mm; }
@media print {
  body { max-width: none; padding: 0; }
  table, img, .katex-display { break-inside: avoid; }
  h2 { margin-top: 12mm; }
}
@media screen {
  body { box-shadow: 0 0 4mm rgba(0,0,0,.12); }
}
"""

html = ("<!DOCTYPE html>\n"
        '<html lang="zh-CN">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
"<title>零遥测电网碳强度预测（中文稿 v2）</title>\n"
        + KATEX + "\n<style>\n" + CSS + "\n</style>\n</head>\n<body>\n"
        + body +
        "\n</body>\n</html>\n")

OUT.write_text(html, encoding="utf-8")
size_mb = OUT.stat().st_size / 1e6
print("OK", OUT, f"({size_mb:.2f} MB)")
