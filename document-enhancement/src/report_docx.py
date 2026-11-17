"""Chuyển docs/BAO_CAO_PBL6.md → docs/BAO_CAO_PBL6.docx (python-docx, không cần pandoc).
(Bản sao từ archive/ocr_topic/pbl6_pipeline/n10_docx.py; đường dẫn ảnh tính theo thư mục chứa tệp .md.)

Hỗ trợ đúng phần Markdown mà báo cáo dùng: tiêu đề #/##/###, đoạn, **đậm**, *nghiêng*, `code`,
danh sách gạch đầu dòng và đánh số (lồng nhau), bảng, ảnh + chú thích "*Hình …*", khối code.
Trang A4, lề 2/2/3/2 cm, Times New Roman 13. Mục lục là trường TOC: Word cập nhật khi mở (hoặc chạy
bước render bằng Word để điền sẵn).

  .venv-ocr/Scripts/python.exe -X utf8 src/report_docx.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
FONT, MONO = "Times New Roman", "Consolas"
TEXT_W_CM = 16.0  # 21 − 3 − 2
INDENT_CM = 0.75


def set_font(target, name, size=None, bold=None, italic=None, color=None):
    """Đặt font cho style hoặc run, ghi đè cả font theo theme (heading mặc định dùng theme)."""
    f = target.font
    f.name = name
    if size:
        f.size = Pt(size)
    if bold is not None:
        f.bold = bold
    if italic is not None:
        f.italic = italic
    if color is not None:
        f.color.rgb = RGBColor.from_string(color)
    rpr = target.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts")
        rpr.insert(0, rf)
    for a in ("asciiTheme", "hAnsiTheme", "eastAsiaTheme", "cstheme"):
        rf.attrib.pop(qn("w:" + a), None)
    for a in ("ascii", "hAnsi", "eastAsia", "cs"):
        rf.set(qn("w:" + a), name)


def setup_styles(doc):
    st = doc.styles
    set_font(st["Normal"], FONT, 13)
    pf = st["Normal"].paragraph_format
    pf.space_before, pf.space_after, pf.line_spacing = Pt(0), Pt(6), 1.2
    for lvl, size, before in [(1, 15, 18), (2, 13.5, 12)]:
        s = st[f"Heading {lvl}"]
        set_font(s, FONT, size, bold=True, italic=False, color="000000")
        p = s.paragraph_format
        p.space_before, p.space_after, p.keep_with_next = Pt(before), Pt(6), True
    for name in ("List Bullet", "List Bullet 2", "List Bullet 3"):
        p = st[name].paragraph_format
        p.space_after, p.line_spacing = Pt(3), 1.2
    toc = st["TOC Heading"] if "TOC Heading" in [s.name for s in st] else None
    if toc is not None:
        set_font(toc, FONT, 15, bold=True, color="000000")


INLINE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|(?<![\w*])\*[^*\s][^*]*?\*(?![\w*]))")


def add_inline(p, text, size=None, bold=False, italic=False):
    for tok in INLINE.split(text):
        if not tok:
            continue
        b, i, mono = bold, italic, False
        if tok.startswith("**") and tok.endswith("**") and len(tok) > 4:
            tok, b = tok[2:-2], True
            # nghiêng lồng trong đậm
            for sub in INLINE.split(tok):
                if sub:
                    add_inline(p, sub, size, True, italic)
            continue
        if tok.startswith("`") and tok.endswith("`"):
            tok, mono = tok[1:-1], True
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            tok, i = tok[1:-1], True
        r = p.add_run(tok)
        r.bold, r.italic = b or None, i or None
        if mono:
            set_font(r, MONO, (size or 13) - 2)
        elif size:
            r.font.size = Pt(size)


def shade(el, fill):
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    el.append(shd)


def add_field(p, instr, placeholder=""):
    r = p.add_run()
    for tag, attr in [("w:fldChar", "begin"), ("w:instrText", None), ("w:fldChar", "separate")]:
        e = OxmlElement(tag)
        if attr:
            e.set(qn("w:fldCharType"), attr)
        else:
            e.set(qn("xml:space"), "preserve")
            e.text = instr
        r._r.append(e)
    if placeholder:
        p.add_run(placeholder)
    r2 = p.add_run()
    e = OxmlElement("w:fldChar")
    e.set(qn("w:fldCharType"), "end")
    r2._r.append(e)


def add_table(doc, rows, aligns):
    ncol = len(rows[0])
    # bề rộng cột tính theo số ký tự: tối thiểu = từ dài nhất (không ngắt được), mong muốn = ô dài nhất
    plain = [[re.sub(r"[*`]", "", r[c]) if c < len(r) else "" for c in range(ncol)] for r in rows]
    minw = [max(max((len(w) for w in row[c].split()), default=1) for row in plain) + 3 for c in range(ncol)]
    want = [max(min(max(len(row[c]) for row in plain), 60) + 1.5, minw[c]) for c in range(ncol)]
    avail = TEXT_W_CM / 0.19
    if sum(want) <= avail:
        chars = [w * avail / sum(want) for w in want]
    elif sum(minw) >= avail:
        chars = [w * avail / sum(minw) for w in minw]
    else:
        f = (avail - sum(minw)) / (sum(want) - sum(minw))
        chars = [m + (w - m) * f for m, w in zip(minw, want)]
    widths = [Cm(TEXT_W_CM * ch / avail) for ch in chars]
    t = doc.add_table(rows=len(rows), cols=ncol)
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    mar = OxmlElement("w:tblCellMar")
    for side, v in [("top", 30), ("left", 80), ("bottom", 30), ("right", 80)]:
        e = OxmlElement(f"w:{side}")
        e.set(qn("w:w"), str(v))
        e.set(qn("w:type"), "dxa")
        mar.append(e)
    look = t._tbl.tblPr.find(qn("w:tblLook"))
    if look is not None:
        look.addprevious(mar)  # đúng thứ tự schema: tblCellMar trước tblLook
    else:
        t._tbl.tblPr.append(mar)
    for k, row in enumerate(rows):
        tr = t.rows[k]
        if k == 0:
            trpr = tr._tr.get_or_add_trPr()
            h = OxmlElement("w:tblHeader")
            h.set(qn("w:val"), "true")
            trpr.append(h)
        cant = OxmlElement("w:cantSplit")
        tr._tr.get_or_add_trPr().append(cant)
        for c in range(ncol):
            cell = tr.cells[c]
            cell.width = widths[c]
            p = cell.paragraphs[0]
            pf = p.paragraph_format
            pf.space_after, pf.line_spacing = Pt(0), 1.0
            pf.keep_with_next = len(rows) <= 15 and k < len(rows) - 1
            if aligns[c] == "right":
                p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            elif aligns[c] == "center":
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            add_inline(p, row[c] if c < len(row) else "", size=11, bold=(k == 0))
            if k == 0:
                shade(cell._tc.get_or_add_tcPr(), "E8E8E8")
    grid = t._tbl.tblGrid
    for gc, w in zip(grid.findall(qn("w:gridCol")), widths):
        gc.set(qn("w:w"), str(int(w.twips)))
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_code(doc, lines):
    maxlen = max((len(l) for l in lines), default=1)
    is_cmd = all(not l or l.startswith(("#", ".", "python")) for l in lines)
    size = 9 if is_cmd else min(9.0, (TEXT_W_CM / 2.54 * 72 - 12) / (0.55 * maxlen))
    for k, line in enumerate(lines):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_after, pf.space_before, pf.line_spacing = Pt(0), Pt(0), 1.0
        pf.keep_with_next = k < len(lines) - 1
        shade(p._p.get_or_add_pPr(), "F3F3F1")
        r = p.add_run(line if line else " ")
        set_font(r, MONO, round(size * 2) / 2, color="5A5A5A" if line.startswith("#") else None)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def add_image(doc, path):
    w, h = Image.open(path).size
    width = TEXT_W_CM
    if h / w * width > 13:
        width = 13 * w / h
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.add_run().add_picture(str(path), width=Cm(width))


LIST = re.compile(r"^(\s*)(- |\d+\. )(.*)$")


def convert(md: Path, out: Path):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin = Cm(2), Cm(2), Cm(3), Cm(2)
    setup_styles(doc)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_field(fp, "PAGE", "1")

    lines = md.read_text("utf-8").splitlines()
    stack = []  # [(indent, content_indent)]
    i = 0
    title_done = False
    break_next = False
    while i < len(lines):
        raw = lines[i]
        s = raw.strip()
        ind = len(raw) - len(raw.lstrip(" "))
        if not s:
            i += 1
            continue
        if s.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].strip().startswith("```"):
                j += 1
            add_code(doc, [l[ind:] if l.startswith(" " * ind) else l for l in lines[i + 1:j]])
            i = j + 1
            continue
        if s.startswith("|"):
            block = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i].strip())
                i += 1
            cells = [[c.strip() for c in b.strip("|").split("|")] for b in block]
            sep = cells[1]
            aligns = ["right" if c.endswith(":") and not c.startswith(":") else
                      "center" if c.startswith(":") and c.endswith(":") else "left" for c in sep]
            add_table(doc, [cells[0]] + cells[2:], aligns)
            continue
        m_img = re.match(r"^!\[[^\]]*\]\(([^)]+)\)$", s)
        if m_img:
            add_image(doc, md.parent / m_img.group(1))
            i += 1
            continue
        if s.startswith("*Hình") and s.endswith("*"):
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            add_inline(p, s[1:-1], size=11, italic=True)
            i += 1
            continue
        if s.startswith("#"):
            stack = []
            lvl = len(s) - len(s.lstrip("#"))
            text = s[lvl:].strip()
            if lvl == 1 and not title_done:
                title_done = True
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(24), Pt(12)
                head, _, sub = text.partition(":")
                r = p.add_run(head.strip())
                set_font(r, FONT, 20, bold=True)
                if sub:
                    p2 = doc.add_paragraph()
                    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p2.add_run(sub.strip())
                    set_font(r, FONT, 16, bold=True)
                # đoạn mở đầu (ngày, nguồn số liệu) rồi mục lục
                j = i + 1
                while j < len(lines) and not lines[j].startswith("#"):
                    if lines[j].strip():
                        p = doc.add_paragraph()
                        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                        add_inline(p, lines[j].strip(), size=11, italic=True)
                    j += 1
                p = doc.add_paragraph()
                p.paragraph_format.space_before = Pt(18)
                r = p.add_run("Mục lục")
                set_font(r, FONT, 15, bold=True)
                add_field(doc.add_paragraph(), 'TOC \\o "1-2" \\h \\z \\u',
                          "Mở tệp trong Word và chọn Cập nhật trường (F9) để hiện mục lục.")
                break_next = True  # mục đầu tiên sang trang mới (không dùng ngắt trang rời để tránh trang trống)
                i = j
                continue
            hp = doc.add_heading(text, level=max(1, lvl - 1))
            if break_next:
                hp.paragraph_format.page_break_before, break_next = True, False
            i += 1
            continue
        m = LIST.match(raw)
        if m:
            lead, marker, text = len(m.group(1)), m.group(2), m.group(3)
            while stack and stack[-1][0] > lead:
                stack.pop()
            if not stack or stack[-1][0] < lead:
                stack.append((lead, lead + len(marker)))
            level = len(stack) - 1
            if marker == "- ":
                p = doc.add_paragraph(style=["List Bullet", "List Bullet 2", "List Bullet 3"][min(level, 2)])
                add_inline(p, text)
                p.paragraph_format.keep_with_next = text.endswith(":") or None
            else:
                p = doc.add_paragraph()
                pf = p.paragraph_format
                pf.left_indent = Cm(INDENT_CM * (level + 1))
                pf.first_line_indent = Cm(-INDENT_CM)
                pf.space_after = Pt(3)
                pf.tab_stops.add_tab_stop(Cm(INDENT_CM * (level + 1)))
                p.add_run(marker.strip() + "\t")
                add_inline(p, text)
                p.paragraph_format.keep_with_next = text.endswith(":") or None
            i += 1
            continue
        if ind > 0 and stack:
            level = len([x for x in stack if x[1] <= ind])
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(INDENT_CM * max(level, 1))
            add_inline(p, s)
            p.paragraph_format.keep_with_next = s.endswith(":") or None
            i += 1
            continue
        stack = []
        p = doc.add_paragraph()
        if "`" not in s:
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        add_inline(p, s)
        p.paragraph_format.keep_with_next = s.endswith(":") or None
        i += 1
    doc.core_properties.title = "Báo cáo PBL6: nâng chất lượng ảnh tài liệu"
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    md = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs/BAO_CAO_PBL6.md"
    convert(md, md.with_suffix(".docx"))
