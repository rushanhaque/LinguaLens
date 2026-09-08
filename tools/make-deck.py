"""
make-deck.py — Build the Lemma pitch deck as a 16:9 PDF.

    python tools/make-deck.py

Everything is drawn as vector art in the app's own palette, so the deck stays
sharp on a projector and matches the product it is describing. Kept in the repo
so the deck can be regenerated after the app changes.
"""

import os
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "Lemma-Pitch-Deck.pdf")

# ── Page: 16:9 widescreen ────────────────────────────────────────────────
W, H = 960, 540
M = 62                      # side margin

# ── Palette: the app's own tokens ────────────────────────────────────────
PAPER = HexColor("#F4F1E8")
SURFACE = HexColor("#FCFBF6")
SURFACE2 = HexColor("#EDE8DA")
INK = HexColor("#24241D")
INK2 = HexColor("#6B6A5E")
INK3 = HexColor("#9A978B")
RULE = HexColor("#DFD9C8")
SAGE = HexColor("#5E6E4C")
SAGE_L = HexColor("#7E9163")
SAGE_P = HexColor("#DDE3D0")
CLAY = HexColor("#9A5C42")
INDIGO = HexColor("#4C5A78")
OCHRE = HexColor("#866B2E")
PLUM = HexColor("#74506E")
CAM_BG = HexColor("#20303A")
CAM_CARD = HexColor("#191A14")
CREAM = HexColor("#FBFAF4")

# ── Fonts ────────────────────────────────────────────────────────────────
FONTS = "C:/Windows/Fonts/"
def _reg(name, filename, **kw):
    try:
        pdfmetrics.registerFont(TTFont(name, FONTS + filename, **kw))
        return name
    except Exception:
        return None

SERIF = _reg("DeckSerif", "georgia.ttf") or "Times-Roman"
SERIF_B = _reg("DeckSerifB", "georgiab.ttf") or "Times-Bold"
SANS = _reg("DeckSans", "segoeui.ttf") or "Helvetica"
SANS_SB = _reg("DeckSansSB", "seguisb.ttf") or "Helvetica-Bold"
SANS_B = _reg("DeckSansB", "segoeuib.ttf") or "Helvetica-Bold"
MONO = "Courier"
CJK = _reg("DeckCJK", "YuGothM.ttc", subfontIndex=0)

slide_no = 0
TOTAL = 19


# ── Primitives ───────────────────────────────────────────────────────────
def Y(top):
    """Convert a distance-from-top into reportlab's bottom-up y."""
    return H - top


def wrap(text, font, size, maxw):
    """Greedy word wrap against real glyph widths."""
    words, lines, cur = text.split(), [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        if pdfmetrics.stringWidth(trial, font, size) <= maxw:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def para(c, x, top, text, font=SANS, size=15, leading=23, maxw=None,
         color=INK2, align="l"):
    """Draw wrapped text from a top coordinate; returns the next top."""
    maxw = maxw or (W - 2 * M)
    c.setFont(font, size)
    c.setFillColor(color)
    for line in wrap(text, font, size, maxw):
        if align == "c":
            c.drawCentredString(x, Y(top + size), line)
        else:
            c.drawString(x, Y(top + size), line)
        top += leading
    return top


def slide(c, title=None, kicker=None, dark=False):
    """Start a page: background, running header, optional title."""
    global slide_no
    slide_no += 1
    c.setFillColor(INK if dark else PAPER)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Running foot: wordmark left, slide number right.
    c.setFont(SANS_SB, 9)
    c.setFillColor(HexColor("#6B6A5E") if not dark else HexColor("#8B897C"))
    c.drawString(M, 26, "LEMMA")
    c.setFont(SANS, 9)
    c.drawRightString(W - M, 26, f"{slide_no} / {TOTAL}")
    c.setStrokeColor(RULE if not dark else HexColor("#3A382C"))
    c.setLineWidth(0.7)
    c.line(M, 42, W - M, 42)

    top = 64
    if kicker:
        c.setFont(SANS_B, 10.5)
        c.setFillColor(SAGE if not dark else SAGE_L)
        c.drawString(M, Y(top + 10), kicker.upper())
        top += 26
    if title:
        c.setFont(SERIF_B, 33)
        c.setFillColor(INK if not dark else CREAM)
        for line in wrap(title, SERIF_B, 33, W - 2 * M):
            c.drawString(M, Y(top + 33), line)
            top += 42
        top += 8
    return top


def card(c, x, top, w, h, fill=SURFACE, radius=14, stroke=RULE, lw=0.8):
    c.setFillColor(fill)
    if stroke:
        c.setStrokeColor(stroke)
        c.setLineWidth(lw)
    c.roundRect(x, Y(top + h), w, h, radius, stroke=1 if stroke else 0, fill=1)


def accent_card(c, x, top, w, h, colour, fill=SURFACE):
    """Card with a coloured spine on its left edge."""
    card(c, x, top, w, h, fill=fill)
    c.setFillColor(colour)
    c.roundRect(x, Y(top + h), 4.5, h, 2.2, stroke=0, fill=1)
    c.setFillColor(fill)
    c.rect(x + 3, Y(top + h), 3, h, stroke=0, fill=1)


def bullets(c, x, top, items, size=15, leading=22, gap=13, maxw=None,
            colour=SAGE, body=INK2):
    """Dot-led list; each item may be a string or (bold_lead, rest)."""
    maxw = maxw or (W - 2 * M - 24)
    for item in items:
        c.setFillColor(colour)
        c.circle(x + 4, Y(top + size * 0.62), 3, stroke=0, fill=1)
        if isinstance(item, tuple):
            lead, rest = item
            c.setFont(SANS_B, size)
            c.setFillColor(INK)
            c.drawString(x + 18, Y(top + size), lead)
            used = pdfmetrics.stringWidth(lead, SANS_B, size)
            first = wrap(rest, SANS, size, maxw - used - 4)
            c.setFont(SANS, size)
            c.setFillColor(body)
            if first:
                c.drawString(x + 18 + used + 4, Y(top + size), first[0])
                rest_txt = " ".join(first[1:]) if len(first) > 1 else ""
            else:
                rest_txt = rest
            top += leading
            if rest_txt:
                top = para(c, x + 18, top, rest_txt, SANS, size, leading,
                           maxw, body)
        else:
            top = para(c, x + 18, top, item, SANS, size, leading, maxw, body)
        top += gap
    return top


def table(c, x, top, w, rows, headers, widths, row_h=34, size=12.5):
    """Simple ruled table with a tinted header row."""
    c.setFillColor(SAGE_P)
    c.roundRect(x, Y(top + 30), w, 30, 7, stroke=0, fill=1)
    c.setFont(SANS_B, 10.5)
    c.setFillColor(SAGE)
    cx = x + 14
    for head, cw in zip(headers, widths):
        c.drawString(cx, Y(top + 20), head.upper())
        cx += cw
    top += 30

    for i, row in enumerate(rows):
        if i % 2 == 0:
            c.setFillColor(SURFACE)
            c.rect(x, Y(top + row_h), w, row_h, stroke=0, fill=1)
        cx = x + 14
        for j, (cell, cw) in enumerate(zip(row, widths)):
            font = SANS_B if j == 0 else SANS
            c.setFillColor(INK if j == 0 else INK2)
            c.setFont(font, size)
            for line in wrap(cell, font, size, cw - 18)[:2]:
                c.drawString(cx, Y(top + row_h / 2 + size * 0.36), line)
                break
            cx += cw
        c.setStrokeColor(RULE)
        c.setLineWidth(0.6)
        c.line(x, Y(top + row_h), x + w, Y(top + row_h))
        top += row_h
    return top


def stat(c, x, top, w, value, label, sub=None):
    card(c, x, top, w, 108)
    c.setFont(SERIF_B, 34)
    c.setFillColor(SAGE)
    c.drawString(x + 18, Y(top + 50), value)
    c.setFont(SANS_B, 12)
    c.setFillColor(INK)
    c.drawString(x + 18, Y(top + 72), label)
    if sub:
        c.setFont(SANS, 10.5)
        c.setFillColor(INK3)
        c.drawString(x + 18, Y(top + 90), sub)


def step_chip(c, x, top, w, h, n, title, body, colour=SAGE):
    card(c, x, top, w, h)
    c.setFillColor(colour)
    c.circle(x + 26, Y(top + 30), 13, stroke=0, fill=1)
    c.setFont(SANS_B, 12)
    c.setFillColor(CREAM)
    c.drawCentredString(x + 26, Y(top + 34.5), str(n))
    c.setFont(SANS_B, 13.5)
    c.setFillColor(INK)
    c.drawString(x + 48, Y(top + 35), title)
    para(c, x + 16, top + 50, body, SANS, 11.5, 16, w - 32, INK2)


def arrow(c, x1, y1, x2, y2, colour=SAGE_L, lw=1.6):
    c.setStrokeColor(colour)
    c.setFillColor(colour)
    c.setLineWidth(lw)
    c.line(x1, y1, x2 - 6, y2)
    c.setLineWidth(0)
    p = c.beginPath()
    p.moveTo(x2, y2)
    p.lineTo(x2 - 8, y2 + 4.5)
    p.lineTo(x2 - 8, y2 - 4.5)
    p.close()
    c.drawPath(p, stroke=0, fill=1)


def brand_mark(c, cx, cy, r, colour):
    """The Lemma mark: a lens ring over two lines of text."""
    c.setStrokeColor(colour)
    c.setLineWidth(r * 0.17)
    c.circle(cx, cy, r, stroke=1, fill=0)
    c.setLineCap(1)
    c.setLineWidth(r * 0.17)
    c.line(cx - r * 0.57, cy + r * 0.21, cx + r * 0.57, cy + r * 0.21)
    c.line(cx - r * 0.57, cy - r * 0.21, cx + r * 0.09, cy - r * 0.21)
    c.setLineCap(0)


# ── The app mockup, drawn to match the real interface ────────────────────
def ar_label(c, x, top, word, gender, phon, colour_hex, colour_phrase, en, em_w=0):
    """One floating AR label as the app renders it."""
    w, h = 186, 84
    c.setFillColor(CAM_CARD)
    c.setStrokeColor(HexColor("#4A4A3E"))
    c.setLineWidth(0.7)
    c.roundRect(x, Y(top + h), w, h, 11, stroke=1, fill=1)

    c.setFont(SANS_B, 15)
    c.setFillColor(CREAM)
    c.drawString(x + 12, Y(top + 24), word)
    gw = pdfmetrics.stringWidth(word, SANS_B, 15)
    if gender:
        c.setFillColor(HexColor("#3A3B31"))
        c.roundRect(x + 18 + gw, Y(top + 24), 22, 13, 3.5, stroke=0, fill=1)
        c.setFont(SANS_B, 8)
        c.setFillColor(HexColor("#BACB9C"))
        c.drawCentredString(x + 29 + gw, Y(top + 21), gender)

    c.setFont(MONO, 9)
    c.setFillColor(HexColor("#93918A"))
    c.drawString(x + 12, Y(top + 38), phon)

    c.setFillColor(HexColor(colour_hex))
    c.circle(x + 16, Y(top + 51), 4.2, stroke=0, fill=1)
    c.setFont(SANS_SB, 10)
    c.setFillColor(HexColor("#D8D5CA"))
    c.drawString(x + 25, Y(top + 54), colour_phrase)

    c.setStrokeColor(HexColor("#3C3C31"))
    c.setLineWidth(0.7)
    c.line(x + 12, Y(top + 62), x + w - 12, Y(top + 62))
    c.setFont(SANS, 9.5)
    c.setFillColor(HexColor("#8B8981"))
    c.drawString(x + 12, Y(top + 76), en)


def corner_ticks(c, x, top, w, h, colour=HexColor("#BACB9C"), L=16):
    c.setStrokeColor(colour)
    c.setLineWidth(2.2)
    c.setLineCap(1)
    y0, y1 = Y(top + h), Y(top)
    for (px, py, dx, dy) in [(x, y1, 1, -1), (x + w, y1, -1, -1),
                             (x, y0, 1, 1), (x + w, y0, -1, 1)]:
        c.line(px, py, px + L * dx, py)
        c.line(px, py, px, py + L * dy)
    c.setLineCap(0)


def app_mockup(c, x, top, w, h):
    """A faithful redraw of the camera screen."""
    c.saveState()
    p = c.beginPath()
    p.roundRect(x, Y(top + h), w, h, 16)
    c.clipPath(p, stroke=0, fill=0)

    c.setFillColor(CAM_BG)
    c.rect(x, Y(top + h), w, h, stroke=0, fill=1)

    # Objects in frame
    c.setFillColor(HexColor("#C0483C"))
    c.rect(x + 34, Y(top + h - 26), 104, 152, stroke=0, fill=1)
    corner_ticks(c, x + 34, top + h - 178, 104, 152)

    c.setFillColor(HexColor("#CFAE45"))
    c.rect(x + 214, Y(top + 118), 62, 62, stroke=0, fill=1)
    corner_ticks(c, x + 214, top + 56, 62, 62)

    c.setFillColor(HexColor("#5478A6"))
    c.rect(x + 330, Y(top + 214), 120, 90, stroke=0, fill=1)
    corner_ticks(c, x + 330, top + 124, 120, 90)

    # Top chrome
    c.setFillColor(HexColor("#22231B"))
    c.roundRect(x + 16, Y(top + 42), 104, 26, 13, stroke=0, fill=1)
    c.setFont(SANS_B, 10.5)
    c.setFillColor(CREAM)
    c.drawString(x + 30, Y(top + 34), "ES  Spanish")

    c.setFillColor(HexColor("#22231B"))
    c.roundRect(x + 16, Y(top + 76), 96, 20, 10, stroke=0, fill=1)
    c.setFillColor(HexColor("#8FA97A"))
    c.circle(x + 27, Y(top + 66), 3.4, stroke=0, fill=1)
    c.setFont(SANS, 9)
    c.setFillColor(HexColor("#A9A79C"))
    c.drawString(x + 35, Y(top + 69), "3 live  ·  51 fps")

    # Labels
    ar_label(c, x + 150, top + 62, "manzana", "la", "man-SA-na",
             "#CFAE45", "la manzana amarilla", "apple")
    ar_label(c, x + 300, top + 116, "coche", "el", "KO-cheh",
             "#C0483C", "el coche rojo", "car")

    # Bottom strip + shutter
    for i, (wd, en) in enumerate([("coche", "car"), ("portátil", "laptop")]):
        bx = x + 20 + i * 108
        c.setFillColor(HexColor("#22231B"))
        c.roundRect(bx, Y(top + h - 62), 98, 34, 9, stroke=0, fill=1)
        c.setFont(SANS_B, 11)
        c.setFillColor(CREAM)
        c.drawString(bx + 12, Y(top + h - 76), wd)
        c.setFont(SANS, 8.5)
        c.setFillColor(HexColor("#8B8981"))
        c.drawString(bx + 12, Y(top + h - 64), en)

    c.setStrokeColor(CREAM)
    c.setLineWidth(2.4)
    c.circle(x + w / 2, Y(top + h - 24), 17, stroke=1, fill=0)
    c.setFillColor(CREAM)
    c.circle(x + w / 2, Y(top + h - 24), 13, stroke=0, fill=1)

    c.restoreState()
    c.setStrokeColor(HexColor("#CFC9B6"))
    c.setLineWidth(1)
    c.roundRect(x, Y(top + h), w, h, 16, stroke=1, fill=0)


# ══ Build ════════════════════════════════════════════════════════════════
c = canvas.Canvas(OUT, pagesize=(W, H))
c.setTitle("Lemma - Summer Internship Mini Project")
c.setAuthor("Rushan Haque")
c.setSubject("A camera app that teaches vocabulary in 12 languages")


# ── 1. Cover ─────────────────────────────────────────────────────────────
c.setFillColor(PAPER)
c.rect(0, 0, W, H, stroke=0, fill=1)
c.setFillColor(SAGE)
c.rect(0, 0, 12, H, stroke=0, fill=1)

icon = os.path.join(ROOT, "icon-512.png")
if os.path.exists(icon):
    c.drawImage(ImageReader(icon), M, Y(196), 104, 104, mask="auto")

c.setFont(SERIF_B, 74)
c.setFillColor(INK)
c.drawString(M, Y(300), "Lemma")
c.setFont(SERIF, 25)
c.setFillColor(SAGE)
c.drawString(M, Y(336), "Look it up by looking.")

c.setStrokeColor(RULE)
c.setLineWidth(1)
c.line(M, Y(364), M + 380, Y(364))

para(c, M, 384, "A camera app that recognises everyday objects and teaches you "
                "their names in 12 languages — with pronunciation, grammar and colour.",
     SANS, 15.5, 24, 560, INK2)

c.setFont(SANS_B, 12)
c.setFillColor(INK)
c.drawString(M, Y(468), "Summer Internship  ·  Mini Project")
c.setFont(SANS, 12)
c.setFillColor(INK2)
c.drawString(M, Y(490), "Rushan Haque   ·   rushanhaque.online")

card(c, W - M - 250, 130, 250, 250, fill=SURFACE)
brand_mark(c, W - M - 125, Y(255), 62, SAGE)
c.setFont(SANS_B, 10)
c.setFillColor(INK3)
c.drawCentredString(W - M - 125, Y(352), "RUNS ENTIRELY ON YOUR PHONE")
slide_no = 1


# ── 2. The problem ───────────────────────────────────────────────────────
c.showPage()
top = slide(c, "Learning words from a list doesn't stick", kicker="The problem")
top = bullets(c, M, top, [
    ("You memorise, then forget. ", "You learn \u201cla voiture = car\u201d on Monday and it is gone by Friday."),
    ("A word list has no context. ", "There is nothing for your brain to attach the word to."),
    ("You learn the wrong words. ", "Apps teach everyone the same 500 words — not the things actually in your room."),
    ("Grammar is left until later. ", "Most apps give you the word but not the gender, the article, or how it changes in a sentence."),
], size=16, leading=24, gap=16, maxw=700)

card(c, M, 402, W - 2 * M, 74, fill=SAGE_P, stroke=None)
c.setFont(SERIF, 19)
c.setFillColor(SAGE)
c.drawCentredString(W / 2, Y(447),
                    "\u201cI know the word for this thing. I just can't remember it when I see the thing.\u201d")


# ── 3. The idea ──────────────────────────────────────────────────────────
c.showPage()
top = slide(c, "So point your camera at it instead", kicker="The idea")
top = para(c, M, top, "You see the object with your own eyes. You learn its name at that "
                      "exact moment. Now your brain has something real to hang the word on.",
           SANS, 17, 26, 780, INK2)
top += 26

steps = [
    (1, "Point", "Aim the camera at anything — a chair, a cup, your dog."),
    (2, "It recognises", "The app works out what the object is, right there on the phone."),
    (3, "It teaches", "The word appears on screen with pronunciation, gender and colour."),
]
cw = (W - 2 * M - 2 * 22) / 3
for i, (n, t, b) in enumerate(steps):
    step_chip(c, M + i * (cw + 22), top, cw, 118, n, t, b)
    if i < 2:
        arrow(c, M + (i + 1) * cw + i * 22 + 5, Y(top + 59),
              M + (i + 1) * (cw + 22) - 5, Y(top + 59))
top += 142
card(c, M, top, W - 2 * M, 56, fill=SURFACE2, stroke=None)
c.setFont(SANS_SB, 14)
c.setFillColor(INK)
c.drawCentredString(W / 2, Y(top + 34),
                    "No typing. No searching. No internet needed. Nothing leaves your phone.")


# ── 4. What it looks like ────────────────────────────────────────────────
c.showPage()
top = slide(c, "What it looks like", kicker="The app")
app_mockup(c, M, top, 470, 300)

bx = M + 500
c.setFont(SANS_B, 14)
c.setFillColor(INK)
c.drawString(bx, Y(top + 14), "Everything on one screen")
bullets(c, bx, top + 34, [
    "The word in big type, with its gender.",
    "How to say it, written simply.",
    "The colour of the object, in a full phrase.",
    "The English word underneath.",
    "Green brackets show what the app has locked on to.",
], size=12.5, leading=18, gap=9, maxw=300)

card(c, bx, top + 216, 336, 84, fill=SAGE_P, stroke=None)
c.setFont(SANS_B, 12)
c.setFillColor(SAGE)
c.drawString(bx + 16, Y(top + 240), "Tap any label")
para(c, bx + 16, top + 250, "Opens the full card: example sentences, your progress "
                            "with that word, and the same word in the other 11 languages.",
     SANS, 11, 15.5, 304, SAGE)


# ── 5. The four parts ────────────────────────────────────────────────────
c.showPage()
top = slide(c, "Four parts to the app", kicker="What it does")
parts = [
    ("Camera", SAGE, "Point and learn. Quiz mode hides the answer until you tap. "
                     "Switch camera, torch, zoom. Save a photo with the words on it. "
                     "Or load a picture from your gallery."),
    ("Learn", INDIGO, "21 decks of words. Five ways to be tested: remember it, pick it, "
                      "hear it, fill the gap, type it. Search all 201 words."),
    ("Progress", OCHRE, "Streaks, XP and levels. A 12-week activity chart. "
                        "21 achievements. Progress per language and per topic."),
    ("Settings", CLAY, "12 languages. Light and dark. Six colour themes. "
                       "Detection tuning. Speech speed. Back up and restore everything."),
]
cw = (W - 2 * M - 3 * 18) / 4
for i, (name, colour, body) in enumerate(parts):
    x = M + i * (cw + 18)
    accent_card(c, x, top, cw, 216, colour)
    c.setFont(SANS_B, 17)
    c.setFillColor(INK)
    c.drawString(x + 20, Y(top + 38), name)
    para(c, x + 20, top + 56, body, SANS, 11.5, 17, cw - 38, INK2)


# ── 6. The clever part ───────────────────────────────────────────────────
c.showPage()
top = slide(c, "The part I am most proud of", kicker="Colour + grammar")
top = para(c, M, top, "A camera can read one thing off an object that no word list can: its colour. "
                      "So the app names the colour — and makes it agree with the noun, the way the "
                      "language really works.", SANS, 15.5, 23, 800, INK2)
top += 16

c.setFont(SANS_B, 13)
c.setFillColor(INK)
c.drawString(M, Y(top + 13), "Point at a red car:")
top += 26

examples = [
    ("French", "la voiture rouge", SANS_B),
    ("Spanish", "el coche rojo", SANS_B),
    ("German", "das rote Auto", SANS_B),
    ("Russian", "\u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043c\u0430\u0448\u0438\u043d\u0430", SANS_B),
    ("Japanese", "\u8d64\u3044\u8eca", CJK or SANS_B),
]
cw = (W - 2 * M - 4 * 14) / 5
for i, (lang, phrase, font) in enumerate(examples):
    x = M + i * (cw + 14)
    card(c, x, top, cw, 76)
    c.setFont(SANS_B, 10)
    c.setFillColor(SAGE)
    c.drawString(x + 14, Y(top + 22), lang.upper())
    c.setFont(font, 15 if font != CJK else 17)
    c.setFillColor(INK)
    c.drawString(x + 14, Y(top + 50), phrase)
top += 96

card(c, M, top, W - 2 * M, 88, fill=SAGE_P, stroke=None)
c.setFont(SANS_B, 14)
c.setFillColor(SAGE)
c.drawString(M + 20, Y(top + 28), "Why this matters")
para(c, M + 20, top + 40, "The colour word changes shape depending on the object: rouge stays, "
                          "rojo becomes roja, rot becomes rote. Getting that agreement right is the "
                          "single hardest habit for a learner — and it is the one thing a camera can "
                          "demonstrate directly, on a real object, in front of you.",
     SANS, 12.5, 17, W - 2 * M - 40, SAGE)


# ── 7. How it works ──────────────────────────────────────────────────────
c.showPage()
top = slide(c, "How it works, start to finish", kicker="The pipeline")
top = para(c, M, top, "This whole loop runs about eight times a second, inside the browser, on the phone.",
           SANS, 14.5, 22, 800, INK2)
top += 18

stages = [
    ("Camera", "A frame from\nthe video"),
    ("AI model", "Finds objects\nand their boxes"),
    ("My filter", "Cleans up and\ntracks them"),
    ("Dictionary", "Looks up the\nword"),
    ("Grammar", "Builds the\nphrase"),
    ("Screen", "Draws the\nlabel"),
]
bw = (W - 2 * M - 5 * 16) / 6
for i, (name, body) in enumerate(stages):
    x = M + i * (bw + 16)
    fill = SURFACE if i not in (2, 4) else SAGE_P
    card(c, x, top, bw, 104, fill=fill, stroke=RULE if fill is SURFACE else None)
    c.setFont(SANS_B, 12.5)
    c.setFillColor(SAGE if fill is SAGE_P else INK)
    c.drawCentredString(x + bw / 2, Y(top + 32), name)
    c.setFont(SANS, 10.5)
    c.setFillColor(SAGE if fill is SAGE_P else INK2)
    for j, line in enumerate(body.split("\n")):
        c.drawCentredString(x + bw / 2, Y(top + 56 + j * 15), line)
    if i < 5:
        arrow(c, x + bw + 2, Y(top + 52), x + bw + 14, Y(top + 52))
top += 124

c.setFont(SANS_B, 11)
c.setFillColor(SAGE)
c.drawString(M, Y(top + 11), "SHADED = THE PARTS I WROTE MYSELF")
top += 26

card(c, M, top, W - 2 * M, 72, fill=SURFACE2, stroke=None)
para(c, M + 20, top + 20, "The AI model is a ready-made one from Google. The interesting work is "
                          "everything around it: making its output stable enough to put on screen, and "
                          "turning a bare English label into a correct sentence in twelve languages.",
     SANS, 12.5, 18, W - 2 * M - 40, INK2)


# ── 8. Main technologies ─────────────────────────────────────────────────
c.showPage()
top = slide(c, "What I used — the main pieces", kicker="Technology")
rows = [
    ["TensorFlow.js", "Google's AI library that runs inside a web browser",
     "Running the object-detection model on the phone itself, with no server"],
    ["COCO-SSD", "A ready-trained AI model that knows 80 everyday objects",
     "Finding what is in the picture and where it is"],
    ["JavaScript (ES Modules)", "The language every browser speaks",
     "All the app logic, split into 26 small, single-purpose files"],
    ["HTML & CSS", "The structure and the styling of a web page",
     "The whole interface, the design system, the animations, light and dark themes"],
    ["Canvas API", "A browser tool for drawing shapes and reading pixels",
     "Drawing the brackets around objects, and reading the colour off them"],
    ["Web Speech API", "Text-to-speech built into the browser",
     "Saying every word and sentence out loud in the right accent"],
]
table(c, M, top, W - 2 * M, rows, ["Technology", "What it is", "What I used it for"],
      [200, 250, 386], row_h=42)


# ── 9. Supporting tools ──────────────────────────────────────────────────
c.showPage()
top = slide(c, "What I used — the supporting tools", kicker="Technology")
rows = [
    ["localStorage", "A small storage box the browser gives every website",
     "Remembering your words, streak, settings and review schedule"],
    ["Service Worker", "A script that runs in the background, even offline",
     "Making the app open and work with no internet after the first visit"],
    ["PWA manifest", "A small settings file browsers read",
     "Letting you install Lemma to your home screen like a normal app"],
    ["Node.js", "JavaScript running outside the browser",
     "Running my 95 automated tests, and generating the app icon in code"],
    ["Git", "Version control — a history of every change",
     "Tracking the work, with a clear message explaining each change"],
]
top = table(c, M, top, W - 2 * M, rows, ["Tool", "What it is", "What I used it for"],
            [200, 250, 386], row_h=42)
top += 22
card(c, M, top, W - 2 * M, 62, fill=SAGE_P, stroke=None)
c.setFont(SANS_B, 13)
c.setFillColor(SAGE)
c.drawString(M + 20, Y(top + 26), "No frameworks, no build step, no server")
para(c, M + 20, top + 36, "The app is plain HTML, CSS and JavaScript files. You can put the folder on "
                          "any web host and it runs. That was a deliberate choice: fewer moving parts to break.",
     SANS, 12, 16, W - 2 * M - 40, SAGE)


# ── 10. Making the AI reliable ───────────────────────────────────────────
c.showPage()
top = slide(c, "Making a noisy AI model usable", kicker="The hard part")
top = para(c, M, top, "Straight out of the box the model flickers, mislabels things, and loses track of "
                      "them. Seven steps of my own sit between the model and the screen.",
           SANS, 14.5, 22, 820, INK2)
top += 14

steps = [
    ("1  Shape check", "A short, wide box is a bad \u201cbottle\u201d, whatever the model says."),
    ("2  Size check", "Reject boxes far too big or too small for that kind of object."),
    ("3  Remove duplicates", "Two boxes on the same thing become one."),
    ("4  Break ties", "A phone and a remote cannot both claim the same rectangle."),
    ("5  Match to last frame", "Each object keeps its own identity between frames."),
    ("6  Wait before showing", "A label appears only after several frames agree."),
    ("7  Smooth and predict", "Labels glide, and lead slightly when you pan the phone."),
]
cw = (W - 2 * M - 2 * 16) / 3
for i, (t, b) in enumerate(steps):
    col, row = i % 3, i // 3
    x = M + col * (cw + 16)
    y = top + row * 78
    card(c, x, y, cw, 66)
    c.setFont(SANS_B, 12.5)
    c.setFillColor(SAGE)
    c.drawString(x + 16, Y(y + 24), t)
    para(c, x + 16, y + 32, b, SANS, 10.5, 14.5, cw - 32, INK2)
top += 78 * 3 + 4

card(c, M + 2 * (cw + 16), top - 78, cw, 66, fill=SAGE_P, stroke=None)
c.setFont(SANS_B, 12)
c.setFillColor(SAGE)
c.drawString(M + 2 * (cw + 16) + 16, Y(top - 78 + 24), "The result")
para(c, M + 2 * (cw + 16) + 16, top - 78 + 32,
     "Two cats side by side stay two labels, instead of collapsing into one.",
     SANS, 10.5, 14.5, cw - 32, SAGE)


# ── 11. The grammar engine ───────────────────────────────────────────────
c.showPage()
top = slide(c, "I don't store sentences — I build them", kicker="Grammar engine")
top = para(c, M, top, "Writing every sentence by hand would mean thousands of lines of text to get wrong. "
                      "Instead I store the bare word plus its gender, and a small engine works out the rest.",
           SANS, 14.5, 22, 820, INK2)
top += 20

card(c, M, top, 250, 150)
c.setFont(SANS_B, 12)
c.setFillColor(SAGE)
c.drawString(M + 18, Y(top + 26), "WHAT I STORE")
c.setFont(MONO, 12)
c.setFillColor(INK)
c.drawString(M + 18, Y(top + 56), "apple")
c.drawString(M + 18, Y(top + 78), "  fr: pomme")
c.drawString(M + 18, Y(top + 100), "  gender: feminine")
c.setFont(SANS, 11)
c.setFillColor(INK3)
c.drawString(M + 18, Y(top + 130), "Two facts. That is all.")

arrow(c, M + 264, Y(top + 76), M + 300, Y(top + 76))

card(c, M + 312, top, W - M - (M + 312), 150, fill=SAGE_P, stroke=None)
c.setFont(SANS_B, 12)
c.setFillColor(SAGE)
c.drawString(M + 332, Y(top + 26), "WHAT THE ENGINE WORKS OUT")
outs = [
    "la pomme  ·  une pomme",
    "C'est une pomme.   Je vois une pomme.   O\u00f9 est la pomme ?",
    "l'orange  —  it knows to contract before a vowel",
    "der / die / das, and how German changes in \u201cI see …\u201d",
    "la pomme blanche  —  the adjective agrees, automatically",
]
oy = top + 44
for o in outs:
    c.setFont(SANS, 12.5)
    c.setFillColor(SAGE)
    c.drawString(M + 332, Y(oy + 12), "\u2022  " + o)
    oy += 21
top += 172

card(c, M, top, W - 2 * M, 58, fill=SURFACE2, stroke=None)
c.setFont(SANS_B, 15)
c.setFillColor(INK)
c.drawCentredString(W / 2, Y(top + 26),
                    "201 words  \u00d7  12 languages  \u00d7  5 sentence patterns  =  over 12,000 correct sentences")
c.setFont(SANS, 12)
c.setFillColor(INK2)
c.drawCentredString(W / 2, Y(top + 46), "Not one of them written by hand.")


# ── 12. Spaced repetition ────────────────────────────────────────────────
c.showPage()
top = slide(c, "How it makes the words stick", kicker="Remembering")
top = para(c, M, top, "Finding a word is easy. Keeping it is the hard part. Lemma uses spaced repetition "
                      "— the same idea behind Anki — so you review each word just before you would "
                      "have forgotten it.", SANS, 14.5, 22, 820, INK2)
top += 18

c.setFont(SANS_B, 13)
c.setFillColor(INK)
c.drawString(M, Y(top + 13), "Get it right, and the gap grows:")
top += 26

gaps = ["1 day", "3 days", "1 week", "3 weeks", "2 months"]
bw = 118
for i, g in enumerate(gaps):
    x = M + i * (bw + 16)
    card(c, x, top, bw, 52, fill=SAGE_P, stroke=None)
    c.setFont(SANS_B, 15)
    c.setFillColor(SAGE)
    c.drawCentredString(x + bw / 2, Y(top + 32), g)
    if i < 4:
        arrow(c, x + bw + 2, Y(top + 26), x + bw + 14, Y(top + 26))

x = M + 5 * (bw + 16)
card(c, x, top, 172, 52, fill=HexColor("#F0DED8"), stroke=None)
c.setFont(SANS_B, 12.5)
c.setFillColor(CLAY)
c.drawCentredString(x + 86, Y(top + 26), "Get it wrong?")
c.setFont(SANS, 11)
c.drawCentredString(x + 86, Y(top + 42), "back in 10 minutes")
top += 76

c.setFont(SANS_B, 13)
c.setFillColor(INK)
c.drawString(M, Y(top + 13), "Five ways it tests you, so it never gets boring:")
top += 26
modes = [
    ("Recall", "Say it in your head, then grade yourself."),
    ("Choose", "Pick the right meaning from four."),
    ("Listen", "Hear the word, pick what it means."),
    ("Fill the gap", "Complete a real sentence."),
    ("Type", "Spell it out. Accents forgiven."),
]
cw = (W - 2 * M - 4 * 14) / 5
for i, (t, b) in enumerate(modes):
    x = M + i * (cw + 14)
    card(c, x, top, cw, 78)
    c.setFont(SANS_B, 13)
    c.setFillColor(SAGE)
    c.drawString(x + 14, Y(top + 26), t)
    para(c, x + 14, top + 34, b, SANS, 10.5, 14, cw - 28, INK2)


# ── 13. Privacy ──────────────────────────────────────────────────────────
c.showPage()
top = slide(c, "It never sends anything anywhere", kicker="Privacy & offline")
top = para(c, M, top, "A camera app that uploads what it sees is a privacy problem. Lemma does not have "
                      "one, because there is nothing to upload to.", SANS, 15.5, 23, 820, INK2)
top += 22

facts = [
    ("No server", "There is no back end. The whole app is a folder of files."),
    ("No account", "No sign-up, no email, no password, no tracking."),
    ("Nothing uploaded", "Camera frames are read and thrown away in the same instant."),
    ("Works offline", "After the first visit it runs with the internet switched off."),
]
cw = (W - 2 * M - 3 * 18) / 4
for i, (t, b) in enumerate(facts):
    x = M + i * (cw + 18)
    card(c, x, top, cw, 132)
    c.setFillColor(SAGE)
    c.circle(x + 26, Y(top + 34), 11, stroke=0, fill=1)
    c.setFont(SANS_B, 13)
    c.setFillColor(CREAM)
    c.drawCentredString(x + 26, Y(top + 38.5), "\u2713")
    c.setFont(SANS_B, 14)
    c.setFillColor(INK)
    c.drawString(x + 46, Y(top + 39), t)
    para(c, x + 18, top + 60, b, SANS, 11.5, 16, cw - 36, INK2)
top += 156

card(c, M, top, W - 2 * M, 62, fill=SAGE_P, stroke=None)
para(c, M + 20, top + 18, "This also makes it very cheap to run and very hard to break: no database to "
                          "maintain, no server bill, no user data to lose. For a project that has to keep "
                          "working after the internship ends, that mattered.",
     SANS, 12.5, 17, W - 2 * M - 40, SAGE)


# ── 14. The numbers ──────────────────────────────────────────────────────
c.showPage()
top = slide(c, "The project in numbers", kicker="Scale")
cw = (W - 2 * M - 3 * 18) / 4
stats1 = [("2,412", "Translations", "201 words \u00d7 12 languages"),
          ("12", "Languages", "European, Asian and Arabic"),
          ("21", "Decks", "12 by sight, 9 by topic"),
          ("11", "Colours", "with full grammar agreement")]
for i, (v, l, s) in enumerate(stats1):
    stat(c, M + i * (cw + 18), top, cw, v, l, s)
top += 128
stats2 = [("~9,600", "Lines of code", "across 26 files"),
          ("95", "Automated tests", "all passing"),
          ("0", "Dependencies", "beyond the AI model"),
          ("5", "Review modes", "plus quiz mode in camera")]
for i, (v, l, s) in enumerate(stats2):
    stat(c, M + i * (cw + 18), top, cw, v, l, s)
top += 132

card(c, M, top, W - 2 * M, 58, fill=SURFACE2, stroke=None)
c.setFont(SANS_SB, 13.5)
c.setFillColor(INK2)
c.drawCentredString(W / 2, Y(top + 26),
                    "Built and tested end to end: camera, AI, grammar, review system, design and offline support.")
c.setFont(SANS, 12)
c.drawCentredString(W / 2, Y(top + 46), "Every piece written from scratch except the object-detection model itself.")


# ── 15. Problems I hit ───────────────────────────────────────────────────
c.showPage()
top = slide(c, "Problems I hit, and how I fixed them", kicker="Debugging")
probs = [
    ("Two cats became one label",
     "I was storing objects by their name, so a second cat overwrote the first. "
     "I rewrote it to give every object its own ID and follow it frame to frame."),
    ("Labels lagged when I moved the phone",
     "The AI only runs a few times a second. I now measure each object's speed and "
     "predict where it will be, so the label keeps up."),
    ("Slow phones stuttered",
     "I added a governor that times the AI and automatically lowers how often it runs, "
     "so the screen stays smooth instead of freezing."),
    ("The colour kept flickering red / orange",
     "A colour is now only accepted after three frames in a row agree on it."),
    ("The app broke after an update",
     "The offline cache was mixing old and new files. I made it serve one complete "
     "version at a time and only switch over when the user reloads."),
    ("People deny camera access",
     "The app now explains why it needs the camera before the browser asks, "
     "instead of springing the prompt on you."),
]
cw = (W - 2 * M - 16) / 2
for i, (t, b) in enumerate(probs):
    col, row = i % 2, i // 2
    x = M + col * (cw + 16)
    y = top + row * 104
    accent_card(c, x, y, cw, 92, CLAY)
    c.setFont(SANS_B, 13.5)
    c.setFillColor(INK)
    c.drawString(x + 20, Y(y + 26), t)
    para(c, x + 20, y + 36, b, SANS, 11.5, 16, cw - 40, INK2)


# ── 16. Testing ──────────────────────────────────────────────────────────
c.showPage()
top = slide(c, "How I knew it actually worked", kicker="Testing")
top = para(c, M, top, "Pointing a phone at a chair is a slow way to check a bug is fixed. So I built ways "
                      "to test it without the phone.", SANS, 15, 23, 820, INK2)
top += 22

items = [
    ("95 automated tests", SAGE,
     "They run in a couple of seconds and check the real logic: the review schedule, "
     "the grammar rules in all 12 languages, the object tracker, and that old saved "
     "progress still loads after an update."),
    ("A fake camera and a fake AI", INDIGO,
     "I replaced the camera with a drawing of coloured blocks, and the AI with a stub "
     "that reports where they are. That let me test the whole pipeline instantly, and "
     "check colour detection with colours I chose."),
    ("Checked by hand in the browser", OCHRE,
     "All 12 languages on screen, light and dark themes, phone and desktop layouts, "
     "and what happens when the camera is blocked or the model fails to download."),
]
cw = (W - 2 * M - 2 * 18) / 3
for i, (t, colour, b) in enumerate(items):
    x = M + i * (cw + 18)
    accent_card(c, x, top, cw, 200, colour)
    c.setFont(SANS_B, 15)
    c.setFillColor(INK)
    c.drawString(x + 20, Y(top + 32), t)
    para(c, x + 20, top + 50, b, SANS, 12, 17, cw - 40, INK2)
top += 224

card(c, M, top, W - 2 * M, 56, fill=SAGE_P, stroke=None)
c.setFont(SANS_SB, 13)
c.setFillColor(SAGE)
c.drawCentredString(W / 2, Y(top + 34),
                    "Several of the bugs on the previous slide were found by these tests, not by luck.")


# ── 17. What I learned ───────────────────────────────────────────────────
c.showPage()
top = slide(c, "What I learned", kicker="Reflection")
top = bullets(c, M, top, [
    ("Running AI in the browser is realistic now. ",
     "A phone can recognise objects in real time with no server behind it."),
    ("The model is only half the job. ",
     "Raw AI output is too noisy to show a user. Most of my effort went into making it steady enough to trust."),
    ("Design the data, not the feature. ",
     "Because words, languages and colours are all just data, adding a 13th language is a data job, not a rewrite."),
    ("Tests find what clicking around does not. ",
     "A whole class of bugs only appears when old saved data meets new code."),
    ("Privacy is a design decision. ",
     "Choosing to have no server removed a whole category of risk before it existed."),
    ("Small details decide whether people stay. ",
     "Explaining the camera prompt before it appears matters as much as the AI behind it."),
], size=14.5, leading=21, gap=12, maxw=800)


# ── 18. What's next ──────────────────────────────────────────────────────
c.showPage()
top = slide(c, "What I would add next", kicker="Future work")
nexts = [
    ("Plural forms", "So it can say \u201cthree apples\u201d correctly. I left this out rather than "
                     "risk teaching wrong grammar in German, Russian and Arabic."),
    ("More objects", "A second model to go beyond the current 80 objects — ideally naming "
                     "the breed of dog, not just \u201cdog\u201d."),
    ("Classroom mode", "A teacher sets a list of objects; students go and find them. "
                       "Turns the app into a scavenger hunt."),
    ("Better voices", "Some phones have no voice installed for some languages. "
                      "Bundling audio would fix that."),
]
cw = (W - 2 * M - 18) / 2
for i, (t, b) in enumerate(nexts):
    col, row = i % 2, i // 2
    x = M + col * (cw + 18)
    y = top + row * 110
    card(c, x, y, cw, 96)
    c.setFillColor(SAGE_P)
    c.roundRect(x + 18, Y(y + 40), 26, 26, 8, stroke=0, fill=1)
    c.setFont(SANS_B, 14)
    c.setFillColor(SAGE)
    c.drawCentredString(x + 31, Y(y + 33), str(i + 1))
    c.setFont(SANS_B, 15)
    c.setFillColor(INK)
    c.drawString(x + 56, Y(y + 34), t)
    para(c, x + 56, y + 48, b, SANS, 11.5, 16, cw - 76, INK2)
top += 232

card(c, M, top, W - 2 * M, 56, fill=SURFACE2, stroke=None)
c.setFont(SANS_SB, 13)
c.setFillColor(INK2)
c.drawCentredString(W / 2, Y(top + 34),
                    "The app is finished and usable today — these are additions, not missing pieces.")


# ── 19. Thank you ────────────────────────────────────────────────────────
c.showPage()
slide_no += 1
c.setFillColor(INK)
c.rect(0, 0, W, H, stroke=0, fill=1)
c.setFillColor(SAGE)
c.rect(0, 0, 12, H, stroke=0, fill=1)

brand_mark(c, 150, Y(210), 56, SAGE_L)

c.setFont(SERIF_B, 60)
c.setFillColor(CREAM)
c.drawString(250, Y(196), "Thank you")
c.setFont(SERIF, 22)
c.setFillColor(SAGE_L)
c.drawString(252, Y(232), "Look it up by looking.")

c.setStrokeColor(HexColor("#3A382C"))
c.setLineWidth(1)
c.line(250, Y(262), 760, Y(262))

c.setFont(SANS, 14)
c.setFillColor(HexColor("#B5B3A6"))
c.drawString(250, Y(292), "Lemma  \u2014  a camera app that teaches you the words for things")
c.setFont(SANS_B, 14)
c.setFillColor(CREAM)
c.drawString(250, Y(322), "Rushan Haque")
c.setFont(SANS, 13)
c.setFillColor(HexColor("#8FA97A"))
c.drawString(250, Y(344), "rushanhaque.online")

c.setFont(SANS_B, 17)
c.setFillColor(CREAM)
c.drawString(250, Y(408), "Questions?")

c.setFont(SANS, 10)
c.setFillColor(HexColor("#6E6C60"))
c.drawRightString(W - M, 26, "Summer Internship  ·  Mini Project")

c.save()
print("Wrote " + OUT)
