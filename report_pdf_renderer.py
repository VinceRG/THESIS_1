import re
import textwrap
from io import BytesIO

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


# ----------------------------------------------------------------------------
# Page geometry and palette (strictly neutral grays -> prints cleanly in B&W)
# ----------------------------------------------------------------------------
PAGE_WIDTH = 595.28
PAGE_HEIGHT = 841.89
MARGIN = 50
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN
FOOTER_Y = 30

BLACK = '#000000'
INK = '#111111'
MUTED = '#555555'
RULE = '#9a9a9a'
HAIRLINE = '#cfcfcf'
PALE = '#f2f2f2'
ZEBRA = '#f7f7f7'

ORG_NAME = 'Accudetek Health Diagnostics'
NUMERIC_RE = re.compile(r'^[+\-]?\d[\d,]*(\.\d+)?\s*%?$')


def _is_numeric(value):
    text = str(value).strip()
    return bool(text) and bool(NUMERIC_RE.match(text))


def _style_axes(axes, grid_axis='y'):
    """Minimal, print-friendly axes: no top/right frame, light grid."""
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color(INK)
        axes.spines[side].set_linewidth(0.8)
    axes.tick_params(axis='both', colors=MUTED, labelsize=7.5, length=3, width=0.6, pad=3)
    axes.grid(axis=grid_axis, color=HAIRLINE, linewidth=0.6)
    axes.set_axisbelow(True)


class _ReportPdf:
    def __init__(self, pdf, title='Report'):
        self.pdf = pdf
        self.title = str(title)
        self.figure = None
        self.figures = []
        self.y = 0
        self.page_number = 0
        self.new_page()

    # ------------------------------------------------------------------
    # Low-level drawing helpers (all coordinates in points, origin top-left
    # for y via self.y, converted to figure fractions here)
    # ------------------------------------------------------------------
    def _line(self, y, x0=MARGIN, x1=PAGE_WIDTH - MARGIN, color=RULE, lw=0.6):
        self.figure.add_artist(Line2D(
            [x0 / PAGE_WIDTH, x1 / PAGE_WIDTH], [y / PAGE_HEIGHT, y / PAGE_HEIGHT],
            transform=self.figure.transFigure, color=color, linewidth=lw,
            solid_capstyle='butt',
        ))

    def _rect(self, x, y_top, width, height, face, edge='none', lw=0.0, zorder=0):
        self.figure.add_artist(Rectangle(
            (x / PAGE_WIDTH, (y_top - height) / PAGE_HEIGHT),
            width / PAGE_WIDTH, height / PAGE_HEIGHT,
            transform=self.figure.transFigure, facecolor=face, edgecolor=edge,
            linewidth=lw, zorder=zorder,
        ))

    def _text(self, x, y, value, size=9, bold=False, italic=False, color=INK,
              ha='left', va='top', family='DejaVu Sans'):
        self.figure.text(
            x / PAGE_WIDTH, y / PAGE_HEIGHT, value, fontsize=size,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal',
            fontfamily=family, color=color, ha=ha, va=va,
        )

    @staticmethod
    def _wrap(value, width_pts, size, factor=0.585):
        chars = max(5, int(width_pts / (size * factor)))
        return textwrap.wrap(' '.join(str(value).split()), width=chars,
                             break_long_words=True, break_on_hyphens=False) or ['']

    # ------------------------------------------------------------------
    # Page management
    # ------------------------------------------------------------------
    def _save_page(self):
        if self.figure is not None:
            self.figures.append(self.figure)
            self.figure = None

    def new_page(self):
        self._save_page()
        self.page_number += 1
        self.figure = Figure(figsize=(PAGE_WIDTH / 72, PAGE_HEIGHT / 72), facecolor='white')
        if self.page_number == 1:
            self.y = PAGE_HEIGHT - MARGIN
            return
        # Running header: small-caps org name on the left, report title below.
        top = PAGE_HEIGHT - 30
        self._text(MARGIN, top, ORG_NAME.upper(), size=6.5, bold=True, color=MUTED)
        y = top - 11
        for line in self._wrap(self.title, CONTENT_WIDTH, 8.5):
            self._text(MARGIN, y, line, size=8.5, color=INK)
            y -= 11
        self._line(y - 1, color=BLACK, lw=1.0)
        self.y = y - 22

    def ensure_space(self, points):
        if self.y - points < MARGIN:
            self.new_page()
            return True
        return False

    def finish(self):
        self._save_page()
        total = len(self.figures)
        for index, figure in enumerate(self.figures, start=1):
            figure.add_artist(Line2D(
                [MARGIN / PAGE_WIDTH, (PAGE_WIDTH - MARGIN) / PAGE_WIDTH],
                [(FOOTER_Y + 11) / PAGE_HEIGHT] * 2, transform=figure.transFigure,
                color=BLACK, linewidth=0.6,
            ))
            figure.text(MARGIN / PAGE_WIDTH, FOOTER_Y / PAGE_HEIGHT,
                        f'{ORG_NAME}   |   Confidential',
                        fontsize=7, color=MUTED, va='center')
            figure.text((PAGE_WIDTH - MARGIN) / PAGE_WIDTH, FOOTER_Y / PAGE_HEIGHT,
                        f'Page {index} of {total}', fontsize=7, color=INK,
                        fontweight='bold', ha='right', va='center')
            self.pdf.savefig(figure, facecolor='white')
        self.figures = []

    # ------------------------------------------------------------------
    # Text blocks
    # ------------------------------------------------------------------
    def write(self, value, size=9, bold=False, italic=False, indent=0,
              family='DejaVu Sans', gap=0, color=INK, leading=1.45):
        value = ' '.join(str(value).split())
        line_height = size * leading
        width = CONTENT_WIDTH - indent
        factor = 0.62 if family.endswith('Mono') else 0.585
        for line in self._wrap(value, width, size, factor):
            self.ensure_space(line_height)
            self._text(MARGIN + indent, self.y, line, size=size, bold=bold,
                       italic=italic, color=color, family=family)
            self.y -= line_height
        self.y -= gap

    def heading(self, value, size=12.5):
        """Section heading: solid black marker, bold title, hairline underneath."""
        lines = self._wrap(value, CONTENT_WIDTH - 14, size, 0.62)
        needed = len(lines) * size * 1.35 + 28
        self.ensure_space(needed)
        self.y -= 4
        block_height = len(lines) * size * 1.35
        self._rect(MARGIN, self.y, 4, block_height - 2, BLACK)
        for line in lines:
            self._text(MARGIN + 12, self.y, line, size=size, bold=True)
            self.y -= size * 1.35
        self.y -= 3
        self._line(self.y, color=HAIRLINE, lw=0.7)
        self.y -= 11

    def rule(self, color=HAIRLINE, lw=0.6, after=6):
        self._line(self.y, color=color, lw=lw)
        self.y -= after

    # ------------------------------------------------------------------
    # Tables
    # ------------------------------------------------------------------
    @staticmethod
    def _column_widths(headers, rows, keys):
        count = len(keys)
        if count == 1:
            return [CONTENT_WIDTH]
        if count == 2:
            return [CONTENT_WIDTH * 0.38, CONTENT_WIDTH * 0.62]
        weights = []
        for index, key in enumerate(keys):
            longest = max([len(headers[index])] + [len(str(row.get(key, ''))) for row in rows[:60]])
            weights.append(min(30, max(8, longest)))
        total = sum(weights)
        return [CONTENT_WIDTH * weight / total for weight in weights]

    def _layout_row(self, values, widths, size):
        wrapped = [self._wrap(value, widths[i] - 12, size) for i, value in enumerate(values)]
        lines = max(len(cell) for cell in wrapped)
        return wrapped, max(20, lines * size * 1.35 + 9)

    def _draw_row(self, wrapped, widths, size, height, numeric, header=False, zebra=False):
        top = self.y
        if header:
            self._rect(MARGIN, top, CONTENT_WIDTH, height, BLACK)
        elif zebra:
            self._rect(MARGIN, top, CONTENT_WIDTH, height, ZEBRA)
        x = MARGIN
        for index, cell_lines in enumerate(wrapped):
            for line_index, line in enumerate(cell_lines):
                ty = top - 5.5 - line_index * size * 1.35
                if numeric[index]:
                    self._text(x + widths[index] - 6, ty, line, size=size, bold=header,
                               color='white' if header else INK, ha='right')
                else:
                    self._text(x + 6, ty, line, size=size, bold=header,
                               color='white' if header else INK)
            x += widths[index]
        if not header:
            self._line(top - height, color=HAIRLINE, lw=0.5)
        self.y -= height

    def table(self, title, note, rows):
        if title:
            self.heading(title)
        if note:
            self.write(note, size=8.5, color=MUTED, gap=6)
        if not rows:
            self.write('No detail rows are available for this report yet.', size=8.5, italic=True)
            return

        keys = list(rows[0].keys())
        headers = [str(key).replace('_', ' ').title() for key in keys]
        widths = self._column_widths(headers, rows, keys)
        size = 8.0 if len(keys) > 3 else 9.0
        numeric = [all(_is_numeric(row.get(key, '')) for row in rows) for key in keys]

        header_wrapped, header_height = self._layout_row(headers, widths, size)

        def draw_header():
            self._draw_row(header_wrapped, widths, size, header_height, numeric, header=True)

        self.ensure_space(header_height + 40)
        draw_header()
        for index, row in enumerate(rows):
            wrapped, height = self._layout_row([row.get(key, '') for key in keys], widths, size)
            if self.ensure_space(height + 1):
                draw_header()  # repeat header on every page
            self._draw_row(wrapped, widths, size, height, numeric, zebra=index % 2 == 1)
        self._line(self.y, color=BLACK, lw=0.8)
        self.y -= 14

    def summary_table(self, sections):
        """Metric / value list with the explanation set directly under each metric."""
        widths = [CONTENT_WIDTH * 0.40, CONTENT_WIDTH * 0.60]
        header_wrapped, header_height = self._layout_row(['Metric', 'Value'], widths, 8.5)
        self.ensure_space(header_height + 60)
        self._draw_row(header_wrapped, widths, 8.5, header_height, [False, False], header=True)
        for section in sections:
            metric = section.get('metric', '')
            value = section.get('value', '')
            explanation = section.get('explanation')
            wrapped, height = self._layout_row([metric, value], widths, 9.5)
            exp_lines = self._wrap(explanation, CONTENT_WIDTH - 12, 8) if explanation else []
            total = height + (len(exp_lines) * 8 * 1.4 + 6 if exp_lines else 0)
            self.ensure_space(total + 2)
            top = self.y
            self._rect(MARGIN, top, CONTENT_WIDTH, total, PALE if exp_lines else 'white', zorder=0)
            x = MARGIN
            for index, cell_lines in enumerate(wrapped):
                for line_index, line in enumerate(cell_lines):
                    self._text(x + 6, top - 6 - line_index * 9.5 * 1.35, line, size=9.5,
                               bold=(index == 0))
                x += widths[index]
            self.y -= height
            if exp_lines:
                self.y += 2
                for line in exp_lines:
                    self._text(MARGIN + 6, self.y, line, size=8, italic=True, color=MUTED)
                    self.y -= 8 * 1.4
                self.y -= 4
            self._line(self.y, color=RULE, lw=0.6)
        self.y -= 14

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------
    def _chart_axes(self, chart_height):
        bottom = self.y - chart_height
        left = MARGIN + 30
        axes = self.figure.add_axes([
            left / PAGE_WIDTH, bottom / PAGE_HEIGHT,
            (PAGE_WIDTH - MARGIN - left) / PAGE_WIDTH, chart_height / PAGE_HEIGHT,
        ])
        return axes, bottom

    def draw_line_chart(self, chart, title):
        datasets = chart.get('datasets', [])
        labels = chart.get('labels', [])
        if not datasets or not labels:
            return

        chart_height = 185
        label_clearance = 52
        self.ensure_space(chart_height + label_clearance + 50)
        self.heading(title)
        self.y -= 16  # room for the legend above the plot
        axes, bottom = self._chart_axes(chart_height)

        line_styles = ['-', '--', ':', '-.', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', 'P']
        shades = ['#000000', '#4d4d4d', '#808080', '#a6a6a6']
        for index, dataset in enumerate(datasets):
            values = dataset.get('data', [])
            axes.plot(
                labels[:len(values)], values,
                color=shades[index % len(shades)],
                linestyle=line_styles[index % len(line_styles)],
                marker=markers[index % len(markers)],
                linewidth=1.6, markersize=4,
                markerfacecolor='white', markeredgewidth=1.1,
                label=str(dataset.get('label') or dataset.get('year', 'Series')),
            )
        axes.set_xlabel(str(chart.get('x_title', 'Period')), color=INK, fontsize=8, labelpad=6)
        axes.set_ylabel('Consultations', color=INK, fontsize=8, labelpad=6)
        _style_axes(axes, 'y')
        axes.set_ylim(bottom=0)
        if len(labels) > 8:
            axes.tick_params(axis='x', labelrotation=35)
            for label in axes.get_xticklabels():
                label.set_horizontalalignment('right')
        axes.margins(x=0.04)
        axes.legend(loc='lower left', bbox_to_anchor=(0, 1.02),
                    ncol=min(4, len(datasets)), frameon=False, fontsize=7.5,
                    handlelength=2.6)
        self.y = bottom - label_clearance

    def draw_bar_chart(self, chart):
        labels = chart.get('labels', [])
        values = chart.get('data', [])
        if not values:
            return
        chart_height = 200
        chart_labels = labels[:len(values)]
        label_width = max(4, min(18, int(150 / max(1, len(chart_labels)))))
        wrapped_labels = [
            '\n'.join(textwrap.wrap(str(label), width=label_width, break_long_words=True,
                                    break_on_hyphens=False))
            for label in chart_labels
        ]
        max_label_lines = max((label.count('\n') + 1 for label in wrapped_labels), default=1)
        label_clearance = max(34, (max_label_lines * 9) + 18)
        self.ensure_space(chart_height + label_clearance)
        axes, bottom = self._chart_axes(chart_height)

        peak = max(range(len(values)), key=values.__getitem__)
        colors = ['#111111' if i == peak else '#a8a8a8' for i in range(len(values))]
        bars = axes.bar(range(len(values)), values, color=colors, edgecolor=INK,
                        linewidth=0.6, width=0.68)
        if len(values) <= 24:
            for bar, value in zip(bars, values):
                axes.annotate(f'{value:g}' if isinstance(value, float) else str(value),
                              (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                              xytext=(0, 2.5), textcoords='offset points',
                              ha='center', va='bottom', fontsize=7, color=INK)
        axes.set_xticks(range(len(wrapped_labels)))
        axes.set_xticklabels(wrapped_labels, rotation=0, ha='center')
        axes.set_ylabel(str(chart.get('y_title', 'Value')), color=INK, fontsize=8, labelpad=6)
        _style_axes(axes, 'y')
        axes.margins(y=0.12)
        axes.set_ylim(bottom=0)
        self.y = bottom - label_clearance


# ----------------------------------------------------------------------------
# Report sections
# ----------------------------------------------------------------------------
def _cover(report, writer, logo_path, prepared_by):
    fig = writer.figure
    # Full-bleed black band along the top edge.
    fig.add_artist(Rectangle((0, (PAGE_HEIGHT - 16) / PAGE_HEIGHT), 1, 16 / PAGE_HEIGHT,
                             transform=fig.transFigure, facecolor=BLACK, edgecolor='none'))

    brand_x = MARGIN
    top = PAGE_HEIGHT - 46
    if logo_path:
        try:
            with Image.open(logo_path) as logo:
                size = 38
                axes = fig.add_axes([MARGIN / PAGE_WIDTH, (top - size) / PAGE_HEIGHT,
                                     size / PAGE_WIDTH, size / PAGE_HEIGHT])
                axes.imshow(logo.convert('L'), cmap='gray')
                axes.axis('off')
                brand_x = MARGIN + size + 12
        except (OSError, ValueError):
            pass
    writer._text(brand_x, top - 4, ORG_NAME.upper(), size=10.5, bold=True)
    writer._text(brand_x, top - 19, 'DIAGNOSTIC REPORT', size=7, color=MUTED)

    writer.y = top - 78
    writer.write(report.get('title', 'Report'), size=23, bold=True, gap=6, leading=1.25)
    writer._line(writer.y + 2, color=BLACK, lw=2.2, x1=MARGIN + 54)
    writer.y -= 14
    if report.get('description'):
        writer.write(report['description'], size=10, color=MUTED, gap=14, leading=1.5)

    sections = report.get('sections', [])
    values = {s.get('metric'): str(s.get('value', '')) for s in sections}
    latest_quarter = values.get('Latest Quarter Total', '').split(':')[0]
    covered = values.get('Data Period Covered') or report.get('historical_period_label')

    metadata = []
    if values.get('Predicted Month'):
        metadata.append(('Forecast period', values['Predicted Month']))
    if values.get('Validation Period'):
        metadata.append(('Validation period', values['Validation Period']))
    if latest_quarter:
        metadata.append(('Latest quarter represented', latest_quarter))
    if covered:
        metadata.append(('Historical data covered', covered))
    if report.get('scope_label'):
        scope = report['scope_label']
        if report.get('date_range_label'):
            scope += f": {report['date_range_label']}"
        metadata.append(('Applied scope', scope))
    branch = report.get('branch_name', '')
    if branch:
        label = branch + (f" ({report['branch_code']})" if report.get('branch_code') else '')
        metadata.append(('Branch', label))
    if report.get('generated_at'):
        metadata.append(('Generated', report['generated_at']))
    if prepared_by:
        metadata.append(('Prepared by', prepared_by))

    _metadata_grid(writer, metadata, signature=bool(prepared_by))
    if report.get('filter_note'):
        writer.write(report['filter_note'], size=8, italic=True, color=MUTED, gap=6)
    writer.y -= 6


def _metadata_grid(writer, items, signature=False):
    """Two-column bordered grid of label / value cells."""
    col_width = CONTENT_WIDTH / 2
    rows = [items[i:i + 2] for i in range(0, len(items), 2)]
    if signature:
        rows.append([('Signature', None)])
    writer.ensure_space(70)
    top_y = writer.y
    writer._line(top_y, color=BLACK, lw=1.0)
    for row in rows:
        is_signature = len(row) == 1 and row[0][1] is None
        cell_lines = [writer._wrap(value, col_width - 16, 9.5) if value else [''] for _, value in row]
        height = 44 if is_signature else 26 + max(len(c) for c in cell_lines) * 9.5 * 1.3
        writer.ensure_space(height)
        top = writer.y
        for index, (label, value) in enumerate(row):
            x = MARGIN + index * col_width + (8 if index else 0)
            writer._text(x, top - 7, label.upper(), size=6.5, bold=True, color=MUTED)
            if is_signature:
                writer._line(top - 36, x0=x, x1=x + col_width - 40, color=INK, lw=0.7)
            else:
                for line_index, line in enumerate(cell_lines[index]):
                    writer._text(x, top - 19 - line_index * 9.5 * 1.3, line, size=9.5)
        if len(row) == 2:
            writer._line(top, x0=MARGIN + col_width, x1=MARGIN + col_width, color=HAIRLINE)
            writer.figure.add_artist(Line2D(
                [(MARGIN + col_width) / PAGE_WIDTH] * 2,
                [(top - height) / PAGE_HEIGHT, top / PAGE_HEIGHT],
                transform=writer.figure.transFigure, color=HAIRLINE, linewidth=0.6))
        writer.y -= height
        writer._line(writer.y, color=HAIRLINE, lw=0.6)
    writer._line(writer.y, color=BLACK, lw=1.0)
    writer.y -= 18


def _write_summary(writer, sections):
    writer.heading('Report Summary')
    writer.summary_table(sections)


def _write_main_details(report, writer):
    title = report.get('details_title', 'Details')
    note = report.get('details_note', '')
    chart = report.get('chart_rows') or {}
    if report.get('key') in {'monthly-consultation', 'quarterly'}:
        if chart.get('datasets'):
            writer.draw_line_chart(chart, title)
            x_title = str(chart.get('x_title', 'the displayed periods')).lower()
            writer.write(
                f'Each line shows consultation counts across {x_title}. Compare line heights '
                'to compare volumes and follow the line to see how counts change between periods.',
                size=8.5, color=MUTED, gap=4)
            labels = chart.get('labels', [])
            for dataset in chart.get('datasets', []):
                values = dataset.get('data', [])
                if values and labels:
                    peak = max(range(min(len(values), len(labels))), key=values.__getitem__)
                    series = dataset.get('label') or dataset.get('year', '')
                    writer.write(f"{series}: highest displayed count is {values[peak]} in {labels[peak]}.",
                                 size=8.5, indent=12)
            writer.y -= 8
        else:
            writer.heading(title)
            writer.write(note, size=8.5, color=MUTED)
            writer.write('No graph data is available yet. Upload consultation records to generate this chart.',
                         size=8.5, italic=True)
        return
    writer.table(title, note, report.get('rows', []))


def _write_extra_charts(charts, writer):
    for chart in charts:
        labels = chart.get('labels', [])
        label_width = max(4, min(18, int(150 / max(1, len(labels)))))
        max_lines = max((len(textwrap.wrap(str(label), width=label_width)) for label in labels), default=1)
        writer.ensure_space(200 + max(34, max_lines * 9 + 18) + 70)
        writer.heading(chart.get('title', 'Chart'))
        writer.y -= 8
        writer.draw_bar_chart(chart)
        y_title = str(chart.get('y_title', 'the displayed measure')).lower()
        writer.write(f'Bars compare {y_title} across categories; taller bars represent higher values. '
                     'The darkest bar marks the highest value.', size=8.5, color=MUTED, gap=2)
        values = chart.get('data', [])
        labels = chart.get('labels', [])
        if values and labels:
            peak = max(range(min(len(values), len(labels))), key=values.__getitem__)
            writer.write(f'The highest displayed value is {values[peak]} for {labels[peak]}.', size=8.5)
        writer.y -= 10


def _write_extra_tables(tables, writer):
    for table in tables:
        rows = table.get('rows', [])
        writer.table(
            table.get('title', 'Additional Details'),
            'Forecast values are grouped using the fields shown in the table.',
            rows,
        )


def build_report_pdf(report, logo_path=None, prepared_by=None):
    output = BytesIO()
    with PdfPages(output) as pdf:
        info = pdf.infodict()
        info['Title'] = str(report.get('title', 'Report'))
        info['Author'] = ORG_NAME
        writer = _ReportPdf(pdf, title=report.get('title', 'Report'))
        _cover(report, writer, logo_path, prepared_by)
        _write_summary(writer, report.get('sections', []))
        _write_main_details(report, writer)
        _write_extra_charts(report.get('extra_charts', []), writer)
        _write_extra_tables(report.get('extra_tables', []), writer)
        writer.finish()
    return output.getvalue()