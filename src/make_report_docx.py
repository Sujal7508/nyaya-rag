"""Convert REPORT.md to REPORT.docx (headings, lists, tables, figures)."""
import re
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches

ROOT = Path(__file__).resolve().parents[1]

def runs(p, text, size=None):
    text = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'\1 (\2)', text)
    for part in re.split(r'(\*\*.+?\*\*|`.+?`|(?<![\w*])\*[^*]+?\*(?![\w*]))', text):
        if not part: continue
        if part.startswith('**'): r = p.add_run(part[2:-2]); r.bold = True
        elif part.startswith('`'): r = p.add_run(part[1:-1]); r.font.name = 'Menlo'; r.font.size = Pt(9.5)
        elif part.startswith('*') and len(part) > 2: r = p.add_run(part[1:-1]); r.italic = True
        else: r = p.add_run(part)
        if size and not part.startswith('`'): r.font.size = Pt(size)

def main():
    s = (ROOT / 'REPORT.md').read_text()
    d = Document(); d.styles['Normal'].font.name = 'Calibri'; d.styles['Normal'].font.size = Pt(11)
    L = s.split('\n'); i = 0
    while i < len(L):
        l = L[i]
        if l.startswith('# '): d.add_heading(l[2:], 0)
        elif l.startswith('## '): d.add_heading(l[3:], 1)
        elif l.startswith('### '): d.add_heading(l[4:], 2)
        elif l.startswith('|'):
            rows = []
            while i < len(L) and L[i].startswith('|'):
                if not re.match(r'^\|[\s\-|]+\|$', L[i]): rows.append([c.strip() for c in L[i].strip('|').split('|')])
                i += 1
            t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = 'Light Grid Accent 1'
            for a, r in enumerate(rows):
                for b, c in enumerate(r):
                    cell = t.cell(a, b); cell.text = ''; runs(cell.paragraphs[0], c, 9.5)
            d.add_paragraph(); continue
        elif l.startswith('!['):
            d.add_picture(str(ROOT / re.search(r'\((.+?)\)', l).group(1)), width=Inches(6.2))
            r = d.add_paragraph().add_run('Figure: ' + re.search(r'\[(.*?)\]', l).group(1)); r.italic = True; r.font.size = Pt(9)
        elif re.match(r'^\s*- ', l): runs(d.add_paragraph(style='List Bullet 2' if l.startswith('    ') else 'List Bullet'), l.strip()[2:])
        elif re.match(r'^\d+\. ', l): runs(d.add_paragraph(style='List Number'), re.sub(r'^\d+\. ', '', l))
        elif l.strip(): runs(d.add_paragraph(), l)
        i += 1
    d.save(ROOT / 'REPORT.docx'); print('saved REPORT.docx', len(d.tables), 'tables', len(d.inline_shapes), 'figures')

if __name__ == '__main__':
    main()
