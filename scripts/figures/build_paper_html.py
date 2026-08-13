#!/usr/bin/env python3
"""Build a self-contained, publication-style HTML page from the paper markdown.

Key fixes vs v1:
  - Images: base64-embedded (no relative path issues, works with file://)
  - Math: protect $...$ and $$...$$ before markdown parsing, restore after,
    so underscores/subscripts are not mangled into <em> tags
  - MathJax 3 with pre-process protection

Usage:
    python scripts/figures/build_paper_html.py
Output:
    docs/paper/transcif_paper.html
"""

import base64
import re
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO / "figures"

import sys
if len(sys.argv) > 1 and sys.argv[1] == "--zh":
    MD_PATH = REPO / "docs/paper/2026-07-26-zeroshot-config-cif-paper-zh.md"
    OUT_PATH = REPO / "docs/paper/transcif_paper_zh.html"
else:
    MD_PATH = REPO / "docs/paper/2026-07-26-zeroshot-config-cif-paper.md"
    OUT_PATH = REPO / "docs/paper/transcif_paper.html"

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".svg": "image/svg+xml"}


def img_to_base64(img_path: Path) -> str:
    """Read an image file and return a data URI."""
    suffix = img_path.suffix.lower()
    mime = MIME.get(suffix, "image/png")
    data = base64.b64encode(img_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def preprocess(md_text: str) -> tuple[str, dict]:
    """Protect math and convert image paths to base64 placeholders.

    Returns (processed_markdown, math_store) where math_store maps
    placeholders back to original LaTeX.
    """
    # --- Step 1: Protect all math from markdown mangling ---
    math_store = {}
    counter = [0]

    def stash_math(match):
        content = match.group(0)
        key = f"MATHJAXPLACEHOLDER{counter[0]}MATHJAXEND"
        math_store[key] = content
        counter[0] += 1
        return key

    # Protect display math first ($$...$$), then inline ($...$)
    md_text = re.sub(r'\$\$[^\$]+\$\$', stash_math, md_text)
    md_text = re.sub(r'\$[^\$\n]+?\$', stash_math, md_text)

    # --- Step 2: Convert <img src="...figures/xxx.png"> to base64 ---
    def replace_img(match):
        full_match = match.group(0)
        src_match = re.search(r'src="([^"]+)"', full_match)
        if not src_match:
            return full_match
        src = src_match.group(1)
        # Normalize path: remove any ../ or leading figures/
        fname = Path(src).name
        img_path = FIG_DIR / fname
        if img_path.exists():
            data_uri = img_to_base64(img_path)
            return full_match.replace(src, data_uri)
        return full_match

    md_text = re.sub(r'<img[^>]+src="[^"]+"[^>]*/?>', replace_img, md_text)

    # --- Step 3: Convert inline figure references like `figures/xxx.png`
    # (used in Chinese version as "图：`figures/xxx.png`") to <img> tags ---
    def inline_fig_to_img(match):
        fname = match.group(1)
        # Extract just the filename
        fname = Path(fname).name
        img_path = FIG_DIR / fname
        if img_path.exists():
            data_uri = img_to_base64(img_path)
            return f'<p align="center"><img src="{data_uri}" width="70%"></p>'
        return match.group(0)

    # Match: 图：`figures/xxx.png`  or  Figure: `figures/xxx.png`
    # Also match patterns like: `figures/xxx.png`、`figures/yyy.png` (Chinese enumeration)
    # Replace each `figures/xxx.png` code span with an embedded image
    md_text = re.sub(r'`figures/([a-zA-Z0-9_./-]+\.png)`',
                     lambda m: (lambda f: f'<p align="center"><img src="{img_to_base64(FIG_DIR / Path(m.group(1)).name)}" width="70%"></p>'
                               if (FIG_DIR / Path(m.group(1)).name).exists() else m.group(0))(m),
                     md_text)

    return md_text, math_store


def restore_math(html: str, math_store: dict) -> str:
    """Restore LaTeX from placeholders in the generated HTML."""
    for key, original in math_store.items():
        html = html.replace(key, original)
    return html


CSS = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666;
  --accent: #2563eb;
  --border: #ddd;
  --code-bg: #f0f0f0;
  --sidebar-bg: #f7f8fa;
  --table-stripe: #f7f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1a1a2e; --fg: #d4d4d4; --muted: #999; --accent: #60a5fa;
    --border: #333; --code-bg: #252535; --sidebar-bg: #161628; --table-stripe: #202030;
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg); color: var(--fg); line-height: 1.75; font-size: 15px;
}
#sidebar {
  position: fixed; top: 0; left: 0; width: 260px; height: 100vh;
  overflow-y: auto; background: var(--sidebar-bg); border-right: 1px solid var(--border);
  padding: 20px 14px; font-size: 13px; z-index: 100;
}
#sidebar h2 { font-size: 12px; text-transform: uppercase; letter-spacing: .5px; color: var(--muted); margin-bottom: 8px; }
#sidebar ul { list-style: none; }
#sidebar li { margin: 1px 0; }
#sidebar a { color: var(--fg); text-decoration: none; display: block; padding: 3px 8px; border-radius: 4px; }
#sidebar a:hover { background: var(--border); color: var(--accent); }
#sidebar .toc-h2 { padding-left: 8px; font-weight: 600; }
#sidebar .toc-h3 { padding-left: 24px; font-size: 12px; color: var(--muted); }
#content { margin-left: 260px; max-width: 820px; padding: 40px 48px 80px; }
h1 { font-size: 25px; line-height: 1.35; margin-bottom: 6px; }
h2 { font-size: 21px; margin-top: 36px; margin-bottom: 10px; padding-bottom: 4px; border-bottom: 2px solid var(--accent); }
h3 { font-size: 17px; margin-top: 26px; margin-bottom: 6px; }
p { margin: 8px 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
strong { font-weight: 700; }
code { background: var(--code-bg); padding: 1px 5px; border-radius: 3px; font-family: 'SF Mono','Fira Code',Consolas,monospace; font-size: .88em; }
pre { background: var(--code-bg); padding: 12px 16px; border-radius: 6px; overflow-x: auto; margin: 10px 0; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 12.5px; }
th, td { border: 1px solid var(--border); padding: 5px 9px; text-align: left; }
th { background: var(--sidebar-bg); font-weight: 700; }
tr:nth-child(even) { background: var(--table-stripe); }
p[align="center"] { margin: 18px auto; text-align: center; }
p[align="center"] img { max-width: 100%; border-radius: 6px; box-shadow: 0 1px 10px rgba(0,0,0,.1); border: 1px solid var(--border); }
p[align="center"] em { display: block; font-size: 12.5px; color: var(--muted); margin-top: 6px; max-width: 92%; margin-left: auto; margin-right: auto; }
hr { border: none; border-top: 1px solid var(--border); margin: 28px 0; }
blockquote { border-left: 3px solid var(--accent); padding-left: 14px; margin: 12px 0; color: var(--muted); }
@media (max-width: 900px) {
  #sidebar { display: none; }
  #content { margin-left: 0; padding: 20px; }
}
@media print {
  #sidebar { display: none; }
  #content { margin: 0; padding: 0; max-width: 100%; }
  p[align="center"] img { box-shadow: none; }
}
"""


def slugify(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s]+', '-', s.strip())


def main():
    md_text = MD_PATH.read_text()
    md_text, math_store = preprocess(md_text)

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists'])
    body_html = md.convert(md_text)

    # Restore protected math
    body_html = restore_math(body_html, math_store)

    # Build TOC from original headings (before math protection doesn't matter for titles)
    orig = MD_PATH.read_text()
    toc_items = []
    for line in orig.split('\n'):
        m = re.match(r'^(#{2,3})\s+(.*)', line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        slug = slugify(title)
        toc_items.append((level, title, slug))

    toc_html = f'<div id="sidebar"><h2>{"目录" if "--zh" in sys.argv else "Contents"}</h2><ul>\n'
    for level, title, slug in toc_items:
        cls = f"toc-h{level}"
        toc_html += f'<li class="{cls}"><a href="#{slug}">{title}</a></li>\n'
    toc_html += '</ul></div>'

    # Add id attributes to h2/h3 headings in body for TOC anchors
    for level, title, slug in toc_items:
        tag = f'h{level}'
        # Find the heading in body_html and add id
        # Markdown may have already added id via toc extension, but we disabled it
        # So let's add manually
        escaped_title = re.escape(title)
        pattern = f'(<{tag}>){escaped_title}'
        body_html = re.sub(pattern, rf'\1 id="{slug}">{title}', body_html, count=1)

    title_match = re.match(r'^#\s+(.+)$', orig, re.MULTILINE)
    title = title_match.group(1) if title_match else "TransCIF Paper"
    is_zh = "--zh" in sys.argv
    lang = "zh-CN" if is_zh else "en"

    html = f"""<!DOCTYPE html>
<html lang="{lang}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$','$'], ['\\\\(','\\\\)']], displayMath: [['$$','$$'], ['\\\\[','\\\\]']] }},
  svg: {{ fontCache: 'global' }},
  startup: {{ typeset: true }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
{toc_html}
<div id="content">
{body_html}
</div>
</body>
</html>"""

    OUT_PATH.write_text(html)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[WRITE] {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"  Math placeholders restored: {len(math_store)}")
    print(f"  Open: open {OUT_PATH}")


if __name__ == "__main__":
    main()
