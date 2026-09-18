#!/usr/bin/env python3
"""Build a self-contained, publication-style HTML page from the paper markdown.

v3 (2026-09-18) — UI redesign:
  - Hero header with metrics band (fd mode), serif display headings
  - Sticky scrollspy sidebar TOC, top progress bar, back-to-top button
  - Carded paper body: rounded figures, tinted tables, soft quote blocks
  - Pipeline unchanged: math protection, base64 images, python-markdown, MathJax 3

Usage:
    python scripts/figures/build_paper_html.py --fd | --zh | (none)
Output:
    docs/paper/transcif_fd_paper_zh.html | transcif_paper_zh.html | transcif_paper.html
"""

import base64
import re
import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent.parent
FIG_DIR = REPO / "figures"

args = sys.argv[1:]
FD_MODE = "--fd" in args
if FD_MODE:
    MD_PATH = REPO / "docs/paper/2026-09-17-transcif-fd-paper-zh.md"
    OUT_PATH = REPO / "docs/paper/transcif_fd_paper_zh.html"
elif "--zh" in args:
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
    """Protect math and convert image paths to base64 placeholders."""
    math_store = {}
    counter = [0]

    def stash_math(match):
        content = match.group(0)
        key = f"MATHJAXPLACEHOLDER{counter[0]}MATHJAXEND"
        math_store[key] = content
        counter[0] += 1
        return key

    md_text = re.sub(r'\$\$[^\$]+\$\$', stash_math, md_text)
    md_text = re.sub(r'\$[^\$\n]+?\$', stash_math, md_text)

    def replace_img(match):
        full_match = match.group(0)
        src_match = re.search(r'src="([^"]+)"', full_match)
        if not src_match:
            return full_match
        src = src_match.group(1)
        fname = Path(src).name
        img_path = FIG_DIR / fname
        if img_path.exists():
            data_uri = img_to_base64(img_path)
            return full_match.replace(src, data_uri)
        return full_match

    md_text = re.sub(r'<img[^>]+src="[^"]+"[^>]*/?>', replace_img, md_text)

    def inline_fig_to_img(match):
        fname = Path(match.group(1)).name
        img_path = FIG_DIR / fname
        if img_path.exists():
            data_uri = img_to_base64(img_path)
            return f'<p align="center"><img src="{data_uri}" width="70%"></p>'
        return match.group(0)

    md_text = re.sub(r'`figures/([a-zA-Z0-9_./-]+\.png)`',
                     lambda m: (lambda f: f'<p align="center"><img src="{img_to_base64(FIG_DIR / Path(m.group(1)).name)}" width="70%"></p>'
                               if (FIG_DIR / Path(m.group(1)).name).exists() else m.group(0))(m),
                     md_text)

    return md_text, math_store

def restore_math(html: str, math_store: dict) -> str:
    for key, original in math_store.items():
        html = html.replace(key, original)
    return html

def slugify(text: str) -> str:
    s = re.sub(r'[^\w\s-]', '', text.lower())
    return re.sub(r'[\s]+', '-', s.strip())

CSS = """
:root{
  --bg:#f5f6f3; --paper:#ffffff; --ink:#20261f; --muted:#6b736c; --line:#e4e7e1;
  --primary:#0a6e4e; --primary-deep:#07412f; --soft:#eaf3ee; --soft-2:#f4f9f6;
  --amber:#b45309; --code-bg:#13211b; --code-ink:#d9e7df; --sel:#bfe3d2;
}
*{box-sizing:border-box;margin:0;padding:0}
html{scroll-behavior:smooth}
::selection{background:var(--sel)}
body{font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Noto Sans SC",
     "Segoe UI",Roboto,"Microsoft YaHei",sans-serif;
     background:var(--bg);color:var(--ink);font-size:16px;line-height:1.9;
     -webkit-font-smoothing:antialiased;}

/* ---- progress bar & back-to-top ---- */
#progress{position:fixed;top:0;left:0;height:3px;width:0;z-index:300;
  background:linear-gradient(90deg,#0a6e4e,#2fa47f);}
#totop{position:fixed;right:28px;bottom:28px;width:44px;height:44px;border-radius:50%;
  background:var(--primary);color:#fff;border:none;cursor:pointer;font-size:17px;
  box-shadow:0 6px 20px rgba(10,110,78,.35);opacity:0;pointer-events:none;
  transition:opacity .25s,background .2s;z-index:200;}
#totop.show{opacity:1;pointer-events:auto}
#totop:hover{background:var(--primary-deep)}

/* ---- hero ---- */
.hero{margin-left:264px;
  background:radial-gradient(1100px 480px at 82% -12%,rgba(47,164,127,.28),transparent 62%),
             linear-gradient(135deg,#0b1f18 0%,#0f3327 55%,#0a4a37 100%);
  color:#f3f7f4;padding:64px 48px 104px;}
.hero-inner{max-width:920px;margin:0 auto;}
.hero-kicker{font-size:12px;letter-spacing:.24em;color:#7fd0ae;margin-bottom:18px;
  font-weight:600;}
.hero h1{font-family:"Songti SC","Noto Serif SC","Source Han Serif SC","Times New Roman",serif;
  font-size:34px;line-height:1.5;font-weight:700;color:#fff;margin-bottom:14px;}
.hero-sub{font-size:15.5px;color:#b9cdc2;line-height:1.8;}
.hero-metrics{display:flex;flex-wrap:wrap;gap:14px;margin-top:34px;}
.hero-metrics .m{flex:1 1 190px;background:rgba(255,255,255,.07);
  border:1px solid rgba(255,255,255,.16);border-radius:12px;padding:16px 18px 14px;}
.hero-metrics b{display:block;font-size:27px;color:#fff;font-weight:700;line-height:1.25;
  font-variant-numeric:tabular-nums;}
.hero-metrics span{font-size:12.5px;color:#a8bfb2;line-height:1.5;display:block;margin-top:3px;}

/* ---- sidebar TOC ---- */
#sidebar{position:fixed;top:0;left:0;width:264px;height:100vh;overflow-y:auto;
  background:#fbfcfa;border-right:1px solid var(--line);
  padding:30px 16px 48px;font-size:13.5px;z-index:100;}
#sidebar h2{font-size:11.5px;letter-spacing:.2em;color:var(--muted);font-weight:600;
  margin:0 0 14px 10px;border:none;padding:0;}
#sidebar ul{list-style:none}
#sidebar li{margin:2px 0}
#sidebar a{display:block;color:#3d4640;text-decoration:none;padding:5px 10px;
  border-radius:6px;border-left:2px solid transparent;line-height:1.55;}
#sidebar a:hover{background:var(--soft);color:var(--primary)}
#sidebar a.active{background:var(--soft);color:var(--primary);font-weight:600;
  border-left-color:var(--primary)}
#sidebar .toc-h3{padding-left:26px;font-size:12.5px;color:var(--muted)}
#sidebar .toc-h3.active{color:var(--primary)}

/* ---- content ---- */
#content{margin-left:264px;padding:0 48px;}
.paper{max-width:880px;margin:-64px auto 80px;background:var(--paper);
  border:1px solid var(--line);border-radius:16px;padding:56px 64px 72px;
  box-shadow:0 1px 2px rgba(20,40,30,.04),0 16px 44px rgba(20,40,30,.07);position:relative;}
#content h1:first-child{display:none}
h2,h3{font-family:"Songti SC","Noto Serif SC","Source Han Serif SC",serif;letter-spacing:.02em;}
h2{font-size:23px;font-weight:700;margin:52px 0 20px;padding-bottom:12px;
  border-bottom:1px solid var(--line);position:relative;line-height:1.5;}
h2:first-child{margin-top:0}
h2::before{content:"";position:absolute;left:0;bottom:-2px;width:64px;height:3px;
  background:var(--primary);border-radius:2px;}
h3{font-size:18.5px;font-weight:700;margin:36px 0 14px;padding-left:14px;
  border-left:4px solid var(--primary);line-height:1.55;}
p{margin:13px 0;text-align:justify;}
a{color:var(--primary);text-decoration:none;border-bottom:1px solid transparent;}
a:hover{border-bottom-color:var(--primary)}
strong{font-weight:700;color:#17402f;}
em{color:inherit}
ul,ol{margin:12px 0;padding-left:1.6em}
li{margin:6px 0}
code{background:#eef2ee;border:1px solid #e0e6e0;padding:1.5px 6px;border-radius:5px;
  font-size:.87em;font-family:"SF Mono",Menlo,Consolas,"JetBrains Mono",monospace;}
pre{background:var(--code-bg);color:var(--code-ink);padding:16px 20px;border-radius:10px;
  overflow-x:auto;margin:18px 0;font-size:13px;line-height:1.75;}
pre code{background:none;border:none;padding:0;color:inherit;font-size:1em;}
blockquote{border-left:4px solid var(--primary);background:var(--soft);
  padding:14px 20px;border-radius:0 10px 10px 0;margin:18px 0;color:#2c3a33;}
blockquote p{margin:8px 0}
hr{border:none;border-top:1px solid var(--line);margin:32px 0;}

/* ---- tables ---- */
table{width:100%;border-collapse:collapse;margin:20px 0;font-size:13px;line-height:1.6;}
th{background:var(--primary-deep);color:#fff;font-weight:600;padding:9px 12px;
  text-align:left;border-bottom:2px solid #05281d;}
td{padding:8px 12px;border-bottom:1px solid var(--line);vertical-align:top;}
tr:nth-child(even) td{background:var(--soft-2);}
tbody tr:hover td{background:var(--soft);}
caption{caption-side:bottom;font-size:12.5px;color:var(--muted);padding-top:8px;}

/* ---- figures ---- */
p[align="center"]{margin:28px auto 10px;text-align:center;}
p[align="center"] img{max-width:100%;height:auto;border-radius:10px;
  border:1px solid var(--line);box-shadow:0 10px 30px rgba(15,50,35,.09);}
p[align="center"] em{display:block;font-size:13px;color:var(--muted);margin-top:12px;
  max-width:92%;margin-left:auto;margin-right:auto;line-height:1.7;text-align:center;}

/* ---- misc ---- */
.MathJax_Display{margin:20px 0!important;}
@media (max-width:1080px){
  #sidebar{display:none}
  .hero{margin-left:0;padding:52px 22px 84px}
  .hero h1{font-size:26px}
  #content{margin-left:0;padding:0 0}
  .paper{margin:-48px 16px 56px;padding:34px 22px 44px;border-radius:12px}
}
@media print{
  #sidebar,#totop,#progress{display:none!important}
  body{background:#fff;font-size:12.5px}
  .hero{background:#fff!important;color:#000;margin:0;padding:20px 0 16px;
        border-bottom:2px solid #222}
  .hero h1{color:#000;font-size:21px;margin-bottom:6px}
  .hero-kicker{color:#555}.hero-sub{color:#444}
  .hero-metrics{display:none}
  #content{margin:0;padding:0}
  .paper{border:none;box-shadow:none;border-radius:0;max-width:100%;
         margin:0;padding:10px 0}
  p[align="center"] img{box-shadow:none}
  a{color:#000}
}
"""

SCRIPT = """
(function(){
  var bar=document.getElementById('progress');
  var toTop=document.getElementById('totop');
  var links=Array.prototype.slice.call(document.querySelectorAll('#sidebar a'));
  var heads=Array.prototype.slice.call(document.querySelectorAll('#content h2[id], #content h3[id]'));
  function onScroll(){
    var h=document.documentElement;
    var max=h.scrollHeight-h.clientHeight;
    if(bar) bar.style.width=(max>0?(h.scrollTop/max*100):0)+'%';
    if(toTop) toTop.classList.toggle('show',h.scrollTop>600);
    var cur='';
    for(var i=0;i<heads.length;i++){
      if(heads[i].getBoundingClientRect().top<=150) cur=heads[i].id;
    }
    for(var j=0;j<links.length;j++){
      var href=links[j].getAttribute('href')||'';
      links[j].classList.toggle('active', href==='#'+cur);
    }
  }
  window.addEventListener('scroll',onScroll,{passive:true});
  window.addEventListener('resize',onScroll);
  onScroll();
  if(toTop) toTop.addEventListener('click',function(){
    window.scrollTo({top:0,behavior:'smooth'});
  });
})();
"""

def build_hero(title: str, zh: bool) -> str:
    if FD_MODE:
        return f"""<header class="hero"><div class="hero-inner">
  <div class="hero-kicker">FD STACK · REVISED 2026-09-17</div>
  <h1>{title}</h1>
  <p class="hero-sub">FuelDecompNet 物理分解 · 信息层级 · 跨洲迁移 —— 零遥测条件下的日前碳强度预测</p>
  <div class="hero-metrics">
    <div class="m"><b>38.4</b><span>零遥测中位 MAE · gCO₂/kWh</span></div>
    <div class="m"><b>29</b><span>真实电力区域 · 3 司法辖区</span></div>
    <div class="m"><b>+6.4</b><span>真实业务天气惩罚 · pooled MAE</span></div>
    <div class="m"><b>4.19×</b><span>事件日最大误差膨胀</span></div>
  </div>
</div></header>"""
    sub = ("Zero-shot cross-region carbon intensity forecasting · "
           "config-only paradigm") if not zh else "零样本跨区域碳强度预测 · config-only 范式"
    return f"""<header class="hero"><div class="hero-inner">
  <div class="hero-kicker">TECHNICAL REPORT</div>
  <h1>{title}</h1>
  <p class="hero-sub">{sub}</p>
</div></header>"""

def main():
    md_text = MD_PATH.read_text()
    md_text, math_store = preprocess(md_text)

    md = markdown.Markdown(extensions=['tables', 'fenced_code', 'sane_lists'])
    body_html = md.convert(md_text)
    body_html = restore_math(body_html, math_store)

    orig = MD_PATH.read_text()
    toc_items = []
    for line in orig.split('\n'):
        m = re.match(r'^(#{2,3})\s+(.*)', line)
        if not m:
            continue
        level = len(m.group(1))
        title_h = m.group(2).strip()
        toc_items.append((level, title_h, slugify(title_h)))

    is_zh = FD_MODE or "--zh" in args
    lang = "zh-CN" if is_zh else "en"
    toc_label = "目录" if is_zh else "Contents"
    toc_html = f'<div id="sidebar"><h2>{toc_label}</h2><ul>\n'
    for level, title_h, slug in toc_items:
        cls = f"toc-h{level}"
        toc_html += f'<li class="{cls}"><a href="#{slug}">{title_h}</a></li>\n'
    toc_html += '</ul></div>'

    for level, title_h, slug in toc_items:
        tag = f'h{level}'
        escaped_title = re.escape(title_h)
        pattern = f'<{tag}>{escaped_title}</{tag}>'
        body_html = re.sub(pattern,
                           lambda m: f'<{tag} id="{slug}">{title_h}</{tag}>',
                           body_html, count=1)

    title_match = re.match(r'^#\s+(.+)$', orig, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else "TransCIF Paper"

    hero = build_hero(title, is_zh)

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
<div id="progress"></div>
{hero}
{toc_html}
<div id="content">
  <div class="paper">
{body_html}
  </div>
</div>
<button id="totop" aria-label="回到顶部">↑</button>
<script>{SCRIPT}</script>
</body>
</html>"""

    OUT_PATH.write_text(html)
    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"[WRITE] {OUT_PATH} ({size_kb:.0f} KB)")
    print(f"  Math placeholders restored: {len(math_store)}")
    print(f"  Open: open {OUT_PATH}")

if __name__ == "__main__":
    main()
