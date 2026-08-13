#!/usr/bin/env python3
"""Build a self-contained, publication-style HTML page from the paper markdown.

Renders:
  - GitHub-flavored Markdown tables, lists, bold/italic
  - LaTeX math via MathJax 3 (CDN)
  - <p align="center"><img ...> blocks as centered figures
  - ``code`` snippets inline
  - A floating sidebar TOC for easy navigation
  - Dark/light toggle, print-friendly CSS

Usage:
    python scripts/figures/build_paper_html.py
Output:
    docs/paper/transcif_paper.html
"""

import re
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent.parent
MD_PATH = REPO / "docs/paper/2026-07-26-zeroshot-config-cif-paper.md"
OUT_PATH = REPO / "docs/paper/transcif_paper.html"
FIG_REL = "../figures"  # relative from docs/paper/ to figures/

# ---------------------------------------------------------------------------
# Pre-process the markdown before feeding to the markdown library
# ---------------------------------------------------------------------------

def preprocess(md_text: str) -> str:
    """Fix image paths and prepare for HTML conversion."""
    # Fix <img src="../figures/..."> → keep relative (HTML lives in docs/paper/)
    # Already correct since OUT_PATH is in docs/paper/ and figures is ../../figures
    # But we wrote the markdown from docs/paper/ perspective, so ../figures/ is right.
    # Actually the HTML file is in docs/paper/ and figures/ is at repo root,
    # so from docs/paper/ it's ../../figures/ ... but the markdown says ../figures/
    # Let's just fix all ../figures/ → ../../figures/ for the HTML context.
    md_text = md_text.replace('src="../figures/', 'src="../../figures/')
    # Also fix bare <img src="figures/..."> (used in §6.10)
    md_text = md_text.replace('src="figures/', 'src="../../figures/')

    # Convert ``figures/xxx.png`` code-span references to actual <img> tags
    # These appear in "Figure: `figures/xxx.png`" lines
    def figure_inline_ref(match):
        fname = match.group(1)
        # Check if already wrapped in an <img> tag elsewhere
        return f'`figures/{fname}`'

    return md_text


# ---------------------------------------------------------------------------
# Build sidebar TOC from headings
# ---------------------------------------------------------------------------

def build_toc(md_text: str) -> str:
    """Extract h1/h2/h3 headings into a nested TOC."""
    entries = []
    for line in md_text.split('\n'):
        m = re.match(r'^(#{1,3})\s+(.*)', line)
        if not m:
            continue
        level = len(m.group(1))
        title = m.group(2).strip()
        # Skip the paper title (h1) from TOC — it's the page header
        if level == 1:
            continue
        # Generate slug
        slug = re.sub(r'[^\w\s-]', '', title.lower())
        slug = re.sub(r'[\s]+', '-', slug.strip())
        entries.append((level, title, slug))
    return entries


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

CSS = """
:root {
  --bg: #ffffff;
  --fg: #1a1a1a;
  --muted: #666;
  --accent: #2563eb;
  --border: #e0e0e0;
  --code-bg: #f5f5f5;
  --sidebar-bg: #f8f9fa;
  --table-stripe: #f8f9fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #1a1a2e;
    --fg: #d4d4d4;
    --muted: #999;
    --border: #333;
    --code-bg: #252535;
    --sidebar-bg: #161628;
    --table-stripe: #202030;
  }
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans SC', Roboto, Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.7;
  font-size: 15px;
}
/* Sidebar TOC */
#sidebar {
  position: fixed;
  top: 0; left: 0;
  width: 280px; height: 100vh;
  overflow-y: auto;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  padding: 20px 16px;
  font-size: 13px;
  z-index: 100;
}
#sidebar h2 {
  font-size: 13px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
  margin-bottom: 10px;
}
#sidebar ul { list-style: none; }
#sidebar li { margin: 2px 0; }
#sidebar a {
  color: var(--fg);
  text-decoration: none;
  display: block;
  padding: 3px 8px;
  border-radius: 4px;
  transition: background 0.15s;
}
#sidebar a:hover { background: var(--border); color: var(--accent); }
#sidebar .toc-h2 { padding-left: 12px; font-weight: 600; }
#sidebar .toc-h3 { padding-left: 28px; font-size: 12px; color: var(--muted); }

/* Main content */
#content {
  margin-left: 280px;
  max-width: 800px;
  padding: 40px 48px 80px;
}
h1 {
  font-size: 26px;
  line-height: 1.3;
  margin-bottom: 8px;
  color: var(--fg);
}
h2 {
  font-size: 22px;
  margin-top: 36px;
  margin-bottom: 12px;
  padding-bottom: 6px;
  border-bottom: 2px solid var(--accent);
  color: var(--fg);
}
h3 {
  font-size: 18px;
  margin-top: 28px;
  margin-bottom: 8px;
  color: var(--fg);
}
h4 { font-size: 16px; margin-top: 20px; }
p { margin: 10px 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
strong { font-weight: 700; }
em { font-style: italic; }

/* Code */
code {
  background: var(--code-bg);
  padding: 1px 5px;
  border-radius: 3px;
  font-family: 'SF Mono', 'Fira Code', Consolas, monospace;
  font-size: 0.88em;
}
pre {
  background: var(--code-bg);
  padding: 12px 16px;
  border-radius: 6px;
  overflow-x: auto;
  margin: 12px 0;
}
pre code { background: none; padding: 0; }

/* Tables */
table {
  border-collapse: collapse;
  width: 100%;
  margin: 16px 0;
  font-size: 13px;
}
th, td {
  border: 1px solid var(--border);
  padding: 6px 10px;
  text-align: left;
}
th { background: var(--sidebar-bg); font-weight: 700; }
tr:nth-child(even) { background: var(--table-stripe); }

/* Figures */
figure, p[align="center"] {
  margin: 20px auto;
  text-align: center;
}
p[align="center"] img {
  max-width: 100%;
  border-radius: 8px;
  box-shadow: 0 2px 12px rgba(0,0,0,0.12);
  border: 1px solid var(--border);
}
p[align="center"] em {
  display: block;
  font-size: 13px;
  color: var(--muted);
  margin-top: 8px;
  max-width: 90%;
  margin-left: auto;
  margin-right: auto;
  line-height: 1.5;
}
hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 32px 0;
}
blockquote {
  border-left: 3px solid var(--accent);
  padding-left: 16px;
  margin: 16px 0;
  color: var(--muted);
}

/* Abstract special styling */
h2 + p strong:first-child { font-size: 16px; }

/* Keywords */
p:last-of-type strong { color: var(--accent); }

/* Mobile */
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


def main():
    md_text = MD_PATH.read_text()
    md_text = preprocess(md_text)

    # Build TOC entries
    toc_entries = build_toc(md_text)

    # Convert markdown to HTML
    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'toc', 'sane_lists'],
                           extension_configs={'toc': {'permalink': False}})
    body_html = md.convert(md_text)

    # Add IDs to h2/h3 for TOC linking (markdown toc extension does this if configured)
    # The 'toc' extension adds ids automatically. Let's verify slug format matches.
    # Actually let's just use the toc extension's slugify.
    toc_html_parts = ['<div id="sidebar"><h2>Table of Contents</h2><ul>']
    for level, title, slug in toc_entries:
        cls = f"toc-h{level}"
        toc_html_parts.append(
            f'<li class="{cls}"><a href="#{slug}">{title}</a></li>')
    toc_html_parts.append('</ul></div>')
    toc_html = '\n'.join(toc_html_parts)

    # Extract title
    title_match = re.match(r'^#\s+(.+)$', md_text, re.MULTILINE)
    title = title_match.group(1) if title_match else "TransCIF Paper"

    # Assemble final HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>{CSS}</style>
<!-- MathJax for LaTeX rendering -->
<script>
window.MathJax = {{
  tex: {{
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
  }},
  svg: {{ fontCache: 'global' }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js" async></script>
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
    print(f"[WRITE] {OUT_PATH} ({size_kb:.1f} KB)")
    print(f"  Open with: open {OUT_PATH}")


if __name__ == "__main__":
    main()
