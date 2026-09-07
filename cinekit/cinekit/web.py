# -*- coding: utf-8 -*-
"""Headless Chromium as a motion-graphics renderer.

CSS, SVG and WebGL give gradient meshes, backdrop blur, variable fonts and
real text shaping — the 2D vocabulary that would otherwise mean After Effects.
The page exposes `window.setFrame(n)` and is stepped deterministically, so the
output is reproducible rather than wall-clock dependent.
"""
import asyncio, os, shutil, tempfile

CHROMIUM = "/opt/pw-browsers/chromium"


class WebScene:
    """Render an HTML page to a PNG sequence, one frame at a time."""

    def __init__(self, html, width=1080, height=1920, scale=1,
                 executable=None, fonts_dir=None):
        self.html, self.W, self.H, self.scale = html, width, height, scale
        self.executable = executable or (CHROMIUM if os.path.exists(CHROMIUM) else None)
        self.fonts_dir = fonts_dir

    def _page_html(self):
        css_fonts = ""
        if self.fonts_dir and os.path.isdir(self.fonts_dir):
            faces = []
            for f in sorted(os.listdir(self.fonts_dir)):
                if f.endswith(".ttf"):
                    stem = os.path.splitext(f)[0]
                    faces.append(f"@font-face{{font-family:'{stem}';"
                                 f"src:url('file://{os.path.join(self.fonts_dir, f)}')"
                                 f" format('truetype');}}")
            css_fonts = "<style>" + "".join(faces) + "</style>"
        return (f"<!doctype html><html><head><meta charset=utf-8>{css_fonts}"
                f"<style>html,body{{margin:0;width:{self.W}px;height:{self.H}px;"
                f"overflow:hidden;background:transparent}}</style></head>"
                f"<body>{self.html}</body></html>")

    async def _run(self, frames, outdir, transparent):
        from playwright.async_api import async_playwright
        os.makedirs(outdir, exist_ok=True)
        tmp = tempfile.mkdtemp()
        page_path = os.path.join(tmp, "scene.html")
        open(page_path, "w", encoding="utf-8").write(self._page_html())
        paths = []
        async with async_playwright() as p:
            kw = dict(args=["--no-sandbox", "--force-color-profile=srgb",
                            "--hide-scrollbars", "--disable-lcd-text",
                            "--allow-file-access-from-files",
                            "--font-render-hinting=none"])
            if self.executable:
                kw["executable_path"] = self.executable
            b = await p.chromium.launch(**kw)
            pg = await b.new_page(viewport={"width": self.W, "height": self.H},
                                  device_scale_factor=self.scale)
            await pg.goto("file://" + page_path)
            await pg.wait_for_timeout(120)
            for n in frames:
                try:
                    await pg.evaluate(f"window.setFrame && window.setFrame({n})")
                except Exception:
                    pass
                fp = os.path.join(outdir, f"{n:05d}.png")
                await pg.screenshot(path=fp, omit_background=transparent)
                paths.append(fp)
            await b.close()
        shutil.rmtree(tmp, ignore_errors=True)
        return paths

    def render(self, frames, outdir, transparent=True):
        return asyncio.run(self._run(list(frames), outdir, transparent))


# --------------------------------------------------------- ready-made scenes
def kinetic_headline(lines, font_family, fps=30, accent="#f0e3c7",
                     bg="transparent", size=118, tracking="-0.02em",
                     stagger=4, rise=90):
    """Words rise into place on a spring — the standard premium title move."""
    spans = "".join(
        f"<div class=line>" + "".join(
            f"<span class=w data-i='{i*8+j}'>{w}</span>"
            for j, w in enumerate(l.split())) + "</div>"
        for i, l in enumerate(lines))
    return f"""
<style>
 body{{background:{bg};display:grid;place-items:center;font-family:'{font_family}',serif}}
 .wrap{{text-align:center;padding:0 60px}}
 .line{{display:block;overflow:hidden;line-height:1.06}}
 .w{{display:inline-block;margin:0 .22em;font-size:{size}px;color:{accent};
     letter-spacing:{tracking};will-change:transform,opacity}}
</style>
<div class=wrap>{spans}</div>
<script>
 const ws=[...document.querySelectorAll('.w')];
 function spring(t,s=170,d=20){{ if(t<=0)return 0; const w0=Math.sqrt(s),z=d/(2*Math.sqrt(s));
   if(z<1){{const wd=w0*Math.sqrt(1-z*z);
     return 1-Math.exp(-z*w0*t)*(Math.cos(wd*t)+z*w0/wd*Math.sin(wd*t));}}
   return 1-Math.exp(-w0*t)*(1+w0*t); }}
 window.setFrame=(n)=>{{
   ws.forEach((el,i)=>{{
     const t=(n-i*{stagger})/{fps};
     const k=spring(Math.max(0,t));
     el.style.transform=`translateY(${{(1-k)*{rise}}}px)`;
     el.style.opacity=Math.min(1,Math.max(0,t*6));
   }});
   return true;
 }};
 window.setFrame(0);
</script>"""


def gradient_mesh(colors=("#7dd3fc", "#f0abfc", "#fde68a"), fps=30, blur=90,
                  speed=0.35):
    """Slow-drifting colour field — the ambient backdrop under product shots."""
    blobs = "".join(f"<div class=b style='background:{c}'></div>" for c in colors)
    return f"""
<style>
 body{{background:#07070b;overflow:hidden}}
 .b{{position:absolute;width:78%;height:52%;border-radius:50%;
     filter:blur({blur}px);opacity:.85;will-change:transform}}
</style>
{blobs}
<script>
 const bs=[...document.querySelectorAll('.b')];
 window.setFrame=(n)=>{{ const t=n/{fps}*{speed};
   bs.forEach((el,i)=>{{ const a=t+i*2.1;
     el.style.transform=`translate(${{Math.sin(a)*36+8}}%,${{Math.cos(a*0.8+i)*40+22}}%)`
                       +` scale(${{1+0.22*Math.sin(a*1.3)}})`; }});
   return true; }};
 window.setFrame(0);
</script>"""
