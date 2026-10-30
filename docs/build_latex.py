#!/usr/bin/env python3
"""
Convert docs/report.md → docs/latex/report.tex + rendered Mermaid diagrams.

Usage:
    python docs/build_latex.py

Output:
    docs/latex/report.tex       LaTeX source (XeLaTeX, 1-column A* CS format)
    docs/latex/images/*.png     Rendered Mermaid diagrams

For PDF: upload docs/latex/ to Overleaf, set compiler to XeLaTeX.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
REPORT_MD = SCRIPT_DIR / "report.md"
OUT_DIR = SCRIPT_DIR / "latex"
IMG_DIR = OUT_DIR / "images"

# ═══════════════════════════════════════════════════════════════════════════
# LaTeX preamble — single-column, A* CS conference style
# ═══════════════════════════════════════════════════════════════════════════

PREAMBLE = r"""\documentclass[12pt,a4paper,onecolumn]{article}

% ── Fonts & Language (XeLaTeX) ──────────────────────────────────
\usepackage{fontspec}
\setmainfont{TeX Gyre Termes}
\setsansfont{TeX Gyre Heros}
\setmonofont{TeX Gyre Cursor}[Scale=0.85]

% ── Math ────────────────────────────────────────────────────────
\usepackage{amsmath,amssymb}

% ── Graphics ────────────────────────────────────────────────────
\usepackage{graphicx}
\usepackage{float}
\usepackage[labelfont=bf,font=small]{caption}
\graphicspath{{images/}}

% ── Tables ──────────────────────────────────────────────────────
\usepackage{booktabs}
\usepackage{array}
\usepackage{tabularx}
\usepackage{makecell}
\renewcommand\theadfont{\bfseries\small}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\newcolumntype{Y}{>{\raggedright\arraybackslash}X}

% ── Code Listings ───────────────────────────────────────────────
\usepackage{listings}
\usepackage{xcolor}
\lstset{
  basicstyle=\small\ttfamily,
  breaklines=true,
  frame=single,
  framesep=3pt,
  rulecolor=\color{gray!40},
  backgroundcolor=\color{gray!5},
  numbers=none,
  showstringspaces=false,
  tabsize=2,
  xleftmargin=6pt,
  xrightmargin=6pt,
  literate={─}{{\textendash}}1 {│}{|}1 {├}{|}1 {└}{|}1,
}
\lstdefinestyle{python}{
  language=Python,
  keywordstyle=\color{blue!70!black}\bfseries,
  stringstyle=\color{green!50!black},
  commentstyle=\color{gray!60}\itshape,
}

% ── Layout ──────────────────────────────────────────────────────
\usepackage[top=2.5cm,bottom=2.5cm,left=2.5cm,right=2.5cm]{geometry}
\usepackage{setspace}
\onehalfspacing
\setlength{\parindent}{1em}
\setlength{\parskip}{0.3em}

% ── Section Formatting ──────────────────────────────────────────
\usepackage{titlesec}
\titleformat{\section}{\Large\bfseries}{\thesection.}{0.5em}{}
\titleformat{\subsection}{\large\bfseries}{\thesubsection}{0.5em}{}
\titleformat{\subsubsection}{\normalsize\bfseries}{\thesubsubsection}{0.5em}{}
\titlespacing*{\section}{0pt}{1.5ex plus 0.5ex}{1ex plus 0.3ex}
\titlespacing*{\subsection}{0pt}{1.2ex plus 0.3ex}{0.8ex plus 0.2ex}

% ── Hyperlinks ──────────────────────────────────────────────────
\usepackage[colorlinks=true,linkcolor=blue!60!black,citecolor=green!50!black,urlcolor=blue!60!black]{hyperref}

% ── TODO markers ────────────────────────────────────────────────
\newcommand{\todofill}[1]{\textcolor{red}{\footnotesize[TODO: #1]}}

% ── Header / Footer ─────────────────────────────────────────────
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\fancyhead[L]{\small\leftmark}
\fancyhead[R]{\small\thepage}
\renewcommand{\headrulewidth}{0.4pt}
\fancypagestyle{plain}{\fancyhf{}\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0pt}}

% ── Lists ───────────────────────────────────────────────────────
\usepackage{enumitem}
\setlist[itemize]{noitemsep,topsep=2pt}
\setlist[enumerate]{noitemsep,topsep=2pt}

"""

# ═══════════════════════════════════════════════════════════════════════════
# LaTeX escaping
# ═══════════════════════════════════════════════════════════════════════════

_LATEX_ESCAPE = {
    '\\': r'\textbackslash{}',
    '&':  r'\&',
    '%':  r'\%',
    '$':  r'\$',
    '#':  r'\#',
    '_':  r'\_',
    '{':  r'\{',
    '}':  r'\}',
    '~':  r'\textasciitilde{}',
    '^':  r'\textasciicircum{}',
}


def _esc_char(ch):
    return _LATEX_ESCAPE.get(ch, ch)


def escape_latex(text):
    """Escape LaTeX-special characters, character by character."""
    return ''.join(_esc_char(c) for c in text)


def escape_texttt(text):
    """Escape for use inside \\texttt{}."""
    return escape_latex(text)


def unescape_md(text):
    r"""Remove markdown backslash-escapes: \< → <, \> → >, etc."""
    return re.sub(r'\\([<>|`*_{}\[\]()#+\-.!])', r'\1', text)

# ═══════════════════════════════════════════════════════════════════════════
# Inline formatting  (markdown → LaTeX)
# ═══════════════════════════════════════════════════════════════════════════

def format_inline(text):
    """Convert inline markdown to LaTeX, preserving math & code."""
    ph = {}
    ctr = [0]

    def _ph(prefix, val):
        key = f"ZZZxPHx{prefix}x{ctr[0]}xZZZ"
        ctr[0] += 1
        ph[key] = val
        return key

    # 1-2) protect math BEFORE unescape — math content is LaTeX, not markdown
    text = re.sub(
        r'\$\$(.+?)\$\$',
        lambda m: _ph('D', m.group(0)),
        text, flags=re.DOTALL,
    )
    text = re.sub(
        r'(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)',
        lambda m: _ph('M', m.group(0)),
        text,
    )

    # 3) now unescape markdown backslash-escapes
    text = unescape_md(text)

    # 4) code spans  `...`
    text = re.sub(
        r'`([^`]+)`',
        lambda m: _ph('C', r'\texttt{' + escape_texttt(m.group(1)) + '}'),
        text,
    )
    # 5) HTML comments  <!-- TODO_FILL: ... -->
    text = re.sub(
        r'<!--\s*(.*?)\s*-->',
        lambda m: _ph('T', r'\todofill{' + escape_latex(m.group(1).strip()) + '}'),
        text,
    )
    # 6) escape remaining text
    text = escape_latex(text)

    # 7) bold  **...**
    text = re.sub(r'\*\*(.+?)\*\*', r'\\textbf{\1}', text)
    # 8) italic  *...*
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)

    # 9) restore placeholders
    for k, v in ph.items():
        text = text.replace(k, v)

    return text

# ═══════════════════════════════════════════════════════════════════════════
# Table conversion
# ═══════════════════════════════════════════════════════════════════════════

def _split_row(line):
    """Split a markdown table row into cells (strip leading/trailing pipes)."""
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return [c.strip() for c in line.split('|')]


def _is_separator(line):
    """Check if line is a table separator row  |---|---|"""
    return bool(re.match(r'^\|?[\s:]*-{2,}[\s:]*(\|[\s:]*-{2,}[\s:]*)*\|?\s*$', line))


def convert_table(lines):
    """Convert markdown table lines to LaTeX tabular/tabularx.

    Auto-detects wide tables and uses tabularx with wrapping Y columns
    for long-content columns so they fit within A4 text width.
    """
    if len(lines) < 2:
        return ''

    header = _split_row(lines[0])
    n_cols = len(header)

    rows = []
    for line in lines[1:]:
        if _is_separator(line):
            continue
        cells = _split_row(line)
        while len(cells) < n_cols:
            cells.append('')
        rows.append(cells[:n_cols])

    # Estimate max content length per column (raw markdown chars)
    max_len = [0] * n_cols
    all_cells = [header] + rows
    for cells in all_cells:
        for j, cell in enumerate(cells):
            max_len[j] = max(max_len[j], len(cell))

    total_len = sum(max_len)

    # Decide layout strategy
    # Wide table: total chars > 90 or any single column > 40 chars
    use_tabularx = (total_len > 90 or any(l > 40 for l in max_len)) and n_cols >= 3

    if use_tabularx:
        # Use Y (raggedright X) for columns with long content, l for short ones
        col_specs = []
        for j in range(n_cols):
            if max_len[j] > 25 and not _looks_numeric(all_cells, j):
                col_specs.append('Y')
            else:
                col_specs.append('l')
        # tabularx needs at least one X/Y column
        if 'Y' not in col_specs:
            # Pick the longest column
            longest = max(range(n_cols), key=lambda j: max_len[j])
            col_specs[longest] = 'Y'
        col_spec = ' '.join(col_specs)
    else:
        col_spec = ' '.join(['l'] * n_cols)

    # Font size for many-column tables
    size_cmd = ''
    if n_cols >= 8:
        size_cmd = r'\scriptsize'
    elif n_cols >= 7:
        size_cmd = r'\footnotesize'
    elif n_cols >= 5:
        size_cmd = r'\small'

    out = []
    out.append('')
    if use_tabularx:
        out.append(r'\noindent')
        if size_cmd:
            out.append('{' + size_cmd)
        out.append(r'\begin{tabularx}{\textwidth}{' + col_spec + '}')
    else:
        out.append(r'\begin{center}')
        if size_cmd:
            out.append(size_cmd)
        out.append(r'\begin{tabular}{' + col_spec + '}')
    out.append(r'\toprule')

    # Header
    hdr_cells = [format_inline(h) for h in header]
    out.append(' & '.join(hdr_cells) + r' \\')
    out.append(r'\midrule')

    # Data rows
    for row in rows:
        data_cells = [format_inline(c) for c in row]
        out.append(' & '.join(data_cells) + r' \\')

    out.append(r'\bottomrule')
    if use_tabularx:
        out.append(r'\end{tabularx}')
        if size_cmd:
            out.append('}')
    else:
        out.append(r'\end{tabular}')
        out.append(r'\end{center}')
    out.append('')

    return '\n'.join(out)


def _looks_numeric(all_cells, col_idx):
    """Check if a column is mostly numeric (skip header row)."""
    numeric_count = 0
    data_rows = all_cells[1:]  # skip header
    for cells in data_rows:
        if col_idx < len(cells):
            val = re.sub(r'[*\s,]', '', cells[col_idx])
            if re.match(r'^-?[\d.]+[%]?$', val) or val in ('—', '–', '-', ''):
                numeric_count += 1
    return len(data_rows) > 0 and numeric_count >= len(data_rows) * 0.7

# ═══════════════════════════════════════════════════════════════════════════
# Mermaid → PNG  (via mermaid-cli / mmdc)
# ═══════════════════════════════════════════════════════════════════════════

def render_mermaid(code, out_png, idx):
    """Render a Mermaid diagram to PNG via mmdc (npx @mermaid-js/mermaid-cli)."""
    tmp_mmd = out_png.with_suffix('.mmd')
    tmp_mmd.write_text(code, encoding='utf-8')

    try:
        import shutil
        npx_cmd = shutil.which('npx') or 'npx.cmd'
        subprocess.run(
            [
                npx_cmd, '-y', '@mermaid-js/mermaid-cli',
                '-i', str(tmp_mmd),
                '-o', str(out_png),
                '-b', 'white',
                '-s', '2',
            ],
            check=True,
            capture_output=True,
            timeout=120,
            shell=(sys.platform == 'win32'),
        )
        print(f"  [OK] diagram_{idx}.png")
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"  [WARN] diagram_{idx}.png failed: {e}")
        return False
    finally:
        if tmp_mmd.exists():
            tmp_mmd.unlink()

# ═══════════════════════════════════════════════════════════════════════════
# Main converter
# ═══════════════════════════════════════════════════════════════════════════

class Converter:
    def __init__(self, md_text, img_dir):
        self.lines = md_text.split('\n')
        self.img_dir = Path(img_dir)
        self.n = len(self.lines)
        self.i = 0
        self.out = []
        self.diagram_idx = 0
        self.mermaid_codes = []

    # ── helpers ──────────────────────────────────────────────────

    def _cur(self):
        return self.lines[self.i] if self.i < self.n else None

    def _peek(self, offset=1):
        j = self.i + offset
        return self.lines[j] if j < self.n else None

    def _emit(self, s):
        self.out.append(s)

    def _skip_blank(self):
        while self.i < self.n and self.lines[self.i].strip() == '':
            self.i += 1

    # ── block detectors ──────────────────────────────────────────

    def _is_header(self, line):
        return line is not None and re.match(r'^#{1,4}\s', line)

    def _is_table_start(self, line):
        if line is None:
            return False
        return line.strip().startswith('|') and '|' in line.strip()[1:]

    def _is_bullet(self, line):
        return line is not None and re.match(r'^[-*]\s', line.strip())

    def _is_numbered(self, line):
        return line is not None and re.match(r'^\d+\.\s', line.strip())

    def _is_code_fence(self, line):
        return line is not None and line.strip().startswith('```')

    def _is_hr(self, line):
        return line is not None and re.match(r'^-{3,}\s*$', line.strip())

    def _is_figure_caption(self, line):
        """Matches *Hình N: ...* pattern (figure caption in italics)."""
        return line is not None and re.match(r'^\*Hình\s+\d+', line.strip())

    # ── block processors ─────────────────────────────────────────

    def _process_header(self):
        line = self._cur()
        m = re.match(r'^(#{1,4})\s+(.*)', line)
        level = len(m.group(1))
        title = m.group(2).strip()

        # Strip manual numbering: "1. " or "1.2 " etc
        title_clean = re.sub(r'^\d+(\.\d+)*\.?\s+', '', title)

        if level == 1:
            self._emit(r'\title{' + format_inline(title_clean) + '}')
            self.i += 1
            return 'title'
        elif level == 2:
            # Special: abstract
            if 'Tóm tắt' in title:
                return 'abstract'
            # Special: references
            if 'Tài liệu tham khảo' in title:
                self._emit('')
                self._emit(r'\section*{Tài liệu tham khảo}')
                self._emit(r'\addcontentsline{toc}{section}{Tài liệu tham khảo}')
                self.i += 1
                return 'references'
            self._emit('')
            self._emit(r'\section{' + format_inline(title_clean) + '}')
        elif level == 3:
            self._emit('')
            self._emit(r'\subsection{' + format_inline(title_clean) + '}')
        elif level == 4:
            self._emit('')
            self._emit(r'\subsubsection{' + format_inline(title_clean) + '}')

        self.i += 1
        return 'section'

    def _process_abstract(self):
        """Collect abstract text until next --- or ## heading."""
        self.i += 1  # skip ## Tóm tắt
        self._skip_blank()
        abstract_lines = []
        while self.i < self.n:
            line = self._cur()
            if line is None:
                break
            if self._is_hr(line) or (self._is_header(line) and line.startswith('## ')):
                break
            # skip HTML comments in abstract
            if line.strip().startswith('<!--'):
                self.i += 1
                continue
            if line.strip():
                abstract_lines.append(format_inline(line.strip()))
            else:
                abstract_lines.append('')
            self.i += 1

        self._emit(r'\begin{abstract}')
        self._emit('\n'.join(abstract_lines))
        self._emit(r'\end{abstract}')

    def _process_metadata(self):
        """Process author/affiliation metadata lines after title."""
        meta = {}
        while self.i < self.n:
            line = self._cur()
            if line is None or self._is_hr(line):
                break
            if line.strip() == '':
                self.i += 1
                continue
            m = re.match(r'^\*\*(.+?):\*\*\s*(.*)', line)
            if m:
                meta[m.group(1).strip()] = m.group(2).strip()
            self.i += 1

        # Build \author block
        parts = []
        if 'Nhóm' in meta:
            parts.append(format_inline(meta['Nhóm']))
        if 'Thành viên' in meta:
            tv = meta['Thành viên']
            if '<!--' in tv:
                tv = re.sub(r'<!--.*?-->', r'\\todofill{danh sách thành viên}', tv)
                parts.append(tv)
            else:
                parts.append(format_inline(tv))
        author_str = r' \\ '.join(parts) if parts else r'\todofill{tác giả}'

        affil_parts = []
        if 'GVHD' in meta:
            gv = meta['GVHD']
            if '<!--' in gv:
                affil_parts.append(r'GVHD: \todofill{tên GVHD}')
            else:
                affil_parts.append(f'GVHD: {format_inline(gv)}')
        if 'Trường' in meta:
            affil_parts.append(format_inline(meta['Trường']))

        self._emit(r'\author{' + author_str)
        if affil_parts:
            self._emit(r'  \\[4pt]')
            self._emit(r'  \normalsize ' + r' \\ '.join(affil_parts))
        self._emit('}')
        self._emit(r'\date{}')

    def _process_table(self):
        """Collect and convert a markdown table."""
        table_lines = []
        while self.i < self.n:
            line = self._cur()
            if line is None or not line.strip().startswith('|'):
                break
            table_lines.append(line)
            self.i += 1
        self._emit(convert_table(table_lines))

    def _process_mermaid(self):
        """Collect mermaid code, render, and emit figure."""
        self.i += 1  # skip ```mermaid
        code_lines = []
        while self.i < self.n:
            line = self._cur()
            if line is not None and line.strip() == '```':
                self.i += 1
                break
            code_lines.append(line)
            self.i += 1

        self.diagram_idx += 1
        code = '\n'.join(code_lines)
        self.mermaid_codes.append(code)
        png_name = f'diagram_{self.diagram_idx}.png'

        # Look ahead for figure caption: *Hình N: ...*
        self._skip_blank()
        caption = ''
        label = f'fig:diagram_{self.diagram_idx}'
        if self.i < self.n and self._is_figure_caption(self._cur()):
            cap_line = self._cur().strip()
            # Remove surrounding *
            cap_line = re.sub(r'^\*', '', cap_line)
            cap_line = re.sub(r'\*$', '', cap_line)
            caption = format_inline(cap_line)
            self.i += 1

        self._emit('')
        self._emit(r'\begin{figure}[H]')
        self._emit(r'\centering')
        self._emit(r'\includegraphics[width=0.78\textwidth,keepaspectratio]{' + png_name + '}')
        if caption:
            self._emit(r'\caption{' + caption + '}')
        self._emit(r'\label{' + label + '}')
        self._emit(r'\end{figure}')
        self._emit('')

    def _process_code_block(self):
        """Process a fenced code block (non-mermaid)."""
        first_line = self._cur().strip()
        lang = ''
        if first_line.startswith('```') and len(first_line) > 3:
            lang = first_line[3:].strip().lower()
        self.i += 1

        code_lines = []
        while self.i < self.n:
            line = self._cur()
            if line is not None and line.strip() == '```':
                self.i += 1
                break
            # Don't escape code lines — lstlisting handles them
            code_lines.append(line if line is not None else '')
            self.i += 1

        style = ''
        if lang == 'python':
            style = ',style=python'

        self._emit('')
        self._emit(r'\begin{lstlisting}[' + style + ']')
        for cl in code_lines:
            self._emit(cl)
        self._emit(r'\end{lstlisting}')
        self._emit('')

    def _process_bullet_list(self):
        """Process a bullet list."""
        self._emit('')
        self._emit(r'\begin{itemize}')
        while self.i < self.n:
            line = self._cur()
            if line is None:
                break
            stripped = line.strip()
            if stripped == '':
                # Check if list continues after blank
                nxt = self._peek()
                if nxt and self._is_bullet(nxt):
                    self.i += 1
                    continue
                break
            if not self._is_bullet(stripped):
                break
            content = re.sub(r'^[-*]\s+', '', stripped)
            self._emit(r'  \item ' + format_inline(content))
            self.i += 1
        self._emit(r'\end{itemize}')
        self._emit('')

    def _process_numbered_list(self):
        """Process a numbered list."""
        self._emit('')
        self._emit(r'\begin{enumerate}')
        while self.i < self.n:
            line = self._cur()
            if line is None:
                break
            stripped = line.strip()
            if stripped == '':
                nxt = self._peek()
                if nxt and self._is_numbered(nxt):
                    self.i += 1
                    continue
                break
            if not self._is_numbered(stripped):
                break
            content = re.sub(r'^\d+\.\s+', '', stripped)
            self._emit(r'  \item ' + format_inline(content))
            self.i += 1
        self._emit(r'\end{enumerate}')
        self._emit('')

    def _process_references(self):
        """Process the references section (simple numbered list)."""
        self.i += 0  # already advanced past heading
        self._emit('')
        self._emit(r'\begin{enumerate}[label={[\arabic*]}]')
        while self.i < self.n:
            line = self._cur()
            if line is None:
                break
            stripped = line.strip()
            if stripped == '':
                self.i += 1
                continue
            if self._is_header(stripped):
                break
            # Reference line: [N] Author, "Title", ...
            m = re.match(r'^\[(\d+)\]\s*(.*)', stripped)
            if m:
                ref_text = m.group(2)
                self._emit(r'  \item ' + format_inline(ref_text))
            self.i += 1
        self._emit(r'\end{enumerate}')

    def _is_md_image(self, line):
        """Matches ![alt](path) markdown image syntax."""
        return line is not None and re.match(r'^!\[.*?\]\(.*?\)\s*$', line.strip())

    def _process_md_image(self):
        """Process a standalone markdown image line: ![alt](path)."""
        line = self._cur().strip()
        m = re.match(r'^!\[(.*?)\]\((.+?)\)\s*$', line)
        if not m:
            self.i += 1
            return
        alt = m.group(1)
        path = m.group(2)
        self.i += 1

        caption = format_inline(alt) if alt else ''
        self._emit('')
        self._emit(r'\begin{figure}[H]')
        self._emit(r'\centering')
        self._emit(r'\includegraphics[width=0.78\textwidth,keepaspectratio]{' + path + '}')
        if caption:
            self._emit(r'\caption{' + caption + '}')
        self._emit(r'\end{figure}')
        self._emit('')

    def _process_paragraph(self):
        """Collect paragraph lines until a block boundary."""
        para_lines = []
        while self.i < self.n:
            line = self._cur()
            if line is None:
                break
            stripped = line.strip()
            if stripped == '':
                self.i += 1
                break
            # Stop at block boundaries
            if (self._is_header(line)
                    or self._is_table_start(line)
                    or self._is_code_fence(line)
                    or self._is_hr(line)
                    or self._is_bullet(stripped)
                    or self._is_numbered(stripped)
                    or self._is_md_image(stripped)):
                break
            para_lines.append(stripped)
            self.i += 1

        if para_lines:
            text = ' '.join(para_lines)
            self._emit('')
            self._emit(format_inline(text))

    # ── main driver ──────────────────────────────────────────────

    def convert(self):
        self.i = 0

        # ── Title ──
        self._skip_blank()
        if self.i < self.n and self.lines[self.i].startswith('# '):
            self._process_header()  # emits \title{...}

        # ── Metadata (author lines) ──
        self._skip_blank()
        self._process_metadata()

        # ── skip HR ──
        if self.i < self.n and self._is_hr(self._cur()):
            self.i += 1

        # ── Build preamble + \begin{document} + \maketitle ──
        preamble = PREAMBLE
        preamble += r'\begin{document}' + '\n'
        preamble += '\n'.join(self.out) + '\n'
        preamble += r'\maketitle' + '\n'

        self.out = []

        # ── Abstract ──
        self._skip_blank()
        if self.i < self.n and 'Tóm tắt' in (self._cur() or ''):
            self._process_abstract()

        # ── skip HR after abstract ──
        self._skip_blank()
        if self.i < self.n and self._is_hr(self._cur()):
            self.i += 1

        # ── Remaining body ──
        in_references = False
        while self.i < self.n:
            line = self._cur()
            if line is None:
                self.i += 1
                continue

            stripped = line.strip()

            # blank line
            if stripped == '':
                self.i += 1
                continue

            # horizontal rule
            if self._is_hr(line):
                self.i += 1
                continue

            # header
            if self._is_header(line):
                result = self._process_header()
                if result == 'abstract':
                    self._process_abstract()
                elif result == 'references':
                    self._process_references()
                continue

            # mermaid code block
            if stripped.startswith('```mermaid'):
                self._process_mermaid()
                continue

            # other code block
            if self._is_code_fence(line):
                self._process_code_block()
                continue

            # table
            if self._is_table_start(line):
                self._process_table()
                continue

            # bullet list
            if self._is_bullet(stripped):
                self._process_bullet_list()
                continue

            # numbered list
            if self._is_numbered(stripped):
                self._process_numbered_list()
                continue

            # markdown image  ![alt](path)
            if self._is_md_image(stripped):
                self._process_md_image()
                continue

            # figure caption without preceding mermaid (standalone italic)
            if self._is_figure_caption(stripped):
                self.i += 1
                continue

            # paragraph
            self._process_paragraph()

        body = '\n'.join(self.out)
        return preamble + body + '\n\n' + r'\end{document}' + '\n'

    def render_diagrams(self):
        """Render all collected Mermaid diagrams to PNG."""
        self.img_dir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for idx, code in enumerate(self.mermaid_codes, 1):
            png = self.img_dir / f'diagram_{idx}.png'
            if render_mermaid(code, png, idx):
                ok += 1
            else:
                # Create placeholder
                _create_placeholder(png, idx)
        return ok

# ═══════════════════════════════════════════════════════════════════════════
# Placeholder image (when mmdc fails)
# ═══════════════════════════════════════════════════════════════════════════

def _create_placeholder(png_path, idx):
    """Create a minimal placeholder PNG (1x1 white pixel) as fallback."""
    import struct, zlib
    def _minimal_png():
        # Minimal 200x40 PNG with text-like content
        # Actually just write a 1x1 white pixel PNG
        sig = b'\x89PNG\r\n\x1a\n'
        # IHDR
        ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
        ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
        ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
        # IDAT
        raw = zlib.compress(b'\x00\xff\xff\xff')
        idat_crc = zlib.crc32(b'IDAT' + raw) & 0xffffffff
        idat = struct.pack('>I', len(raw)) + b'IDAT' + raw + struct.pack('>I', idat_crc)
        # IEND
        iend_crc = zlib.crc32(b'IEND') & 0xffffffff
        iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
        return sig + ihdr + idat + iend

    png_path.write_bytes(_minimal_png())
    print(f"  [PLACEHOLDER] diagram_{idx}.png (mmdc failed, replace manually)")

# ═══════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if not REPORT_MD.exists():
        print(f"Error: {REPORT_MD} not found")
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    md = REPORT_MD.read_text(encoding='utf-8')

    print(f"Converting {REPORT_MD} ...")
    conv = Converter(md, IMG_DIR)
    tex = conv.convert()

    tex_path = OUT_DIR / 'report.tex'
    tex_path.write_text(tex, encoding='utf-8')
    print(f"  [OK] {tex_path}")

    print(f"\nRendering {len(conv.mermaid_codes)} Mermaid diagrams ...")
    ok = conv.render_diagrams()
    print(f"  {ok}/{len(conv.mermaid_codes)} diagrams rendered")

    # ── Try building PDF with XeLaTeX ──
    import shutil
    xelatex = shutil.which('xelatex')
    if xelatex:
        print(f"\nBuilding PDF with XeLaTeX ...")
        for run in range(2):
            result = subprocess.run(
                [xelatex, '-interaction=nonstopmode', '-halt-on-error', 'report.tex'],
                cwd=str(OUT_DIR),
                capture_output=True,
                timeout=120,
            )
            if result.returncode != 0 and run == 1:
                print(f"  [WARN] XeLaTeX returned {result.returncode}")
                log = (OUT_DIR / 'report.log').read_text(encoding='utf-8', errors='replace')
                for line in log.splitlines():
                    if line.startswith('!') or 'Error' in line:
                        print(f"    {line}")
                break
        pdf_path = OUT_DIR / 'report.pdf'
        if pdf_path.exists():
            print(f"  [OK] {pdf_path}")
        else:
            print(f"  [FAIL] PDF not generated — check report.log")
    else:
        print(f"\n  XeLaTeX not found on PATH. Install MiKTeX/TeX Live, or upload to Overleaf.")

    print(f"\n{'='*60}")
    print(f"Output: {OUT_DIR}/")
    print(f"  report.tex          LaTeX source (XeLaTeX)")
    print(f"  images/*.png        Diagram images")
    if xelatex and (OUT_DIR / 'report.pdf').exists():
        print(f"  report.pdf          Compiled PDF")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
