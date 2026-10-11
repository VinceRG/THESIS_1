from io import BytesIO
import textwrap

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image


PAGE_WIDTH = 595.28
PAGE_HEIGHT = 841.89
MARGIN = 48


class _ReportPdf:
    def __init__(self, pdf):
        self.pdf = pdf
        self.figure = None
        self.y = 0
        self.new_page()

    def new_page(self):
        if self.figure is not None:
            self.pdf.savefig(self.figure, facecolor='white')
            self.figure.clear()
        self.figure = Figure(figsize=(PAGE_WIDTH / 72, PAGE_HEIGHT / 72), facecolor='white')
        self.y = PAGE_HEIGHT - MARGIN

    def ensure_space(self, points):
        if self.y - points < MARGIN:
            self.new_page()

    def write(self, value, size=9, bold=False, indent=0, family='DejaVu Sans', gap=0):
        value = ' '.join(str(value).split())
        line_height = size * 1.4
        available_width = PAGE_WIDTH - (MARGIN * 2) - indent
        chars_per_line = max(16, int(available_width / (size * (0.62 if family.endswith('Mono') else 0.52))))
        lines = textwrap.wrap(value, width=chars_per_line, break_long_words=True, break_on_hyphens=False) or ['']
        for line in lines:
            self.ensure_space(line_height)
            self.figure.text(
                (MARGIN + indent) / PAGE_WIDTH,
                self.y / PAGE_HEIGHT,
                line,
                fontsize=size,
                fontweight='bold' if bold else 'normal',
                fontfamily=family,
                color='#111111',
                va='top',
            )
            self.y -= line_height
        self.y -= gap

    def heading(self, value, size=12):
        self.write(value, size=size, bold=True, gap=5)

    def rule(self):
        y = self.y / PAGE_HEIGHT
        self.figure.add_artist(Line2D(
            [MARGIN / PAGE_WIDTH, (PAGE_WIDTH - MARGIN) / PAGE_WIDTH],
            [y, y],
            transform=self.figure.transFigure,
            color='#777777',
            linewidth=0.45,
        ))
        self.y -= 6

    def table(self, title, note, rows):
        if title:
            self.heading(title)
        if note:
            self.write(note, size=8.5, gap=4)
        if not rows:
            self.write('No detail rows are available for this report yet.', size=8.5)
            return

        keys = list(rows[0].keys())
        headers = [str(key).replace('_', ' ').title() for key in keys]
        self.write_table_row(headers, bold=True)
        self.rule()
        for row in rows:
            self.write_table_row([row.get(key, '') for key in keys])
            self.y -= 2

    def write_table_row(self, values, bold=False):
        column_count = max(1, len(values))
        # Keep the composed row within the same printable width used by write().
        # A fixed minimum per column made wide tables exceed the page and overlap.
        row_width = 110
        separator = ' | '
        column_width = max(1, (row_width - (len(separator) * (column_count - 1))) // column_count)
        wrapped = [
            textwrap.wrap(' '.join(str(value).split()), width=column_width,
                          break_long_words=True, break_on_hyphens=False) or ['']
            for value in values
        ]
        for line_index in range(max(len(cell) for cell in wrapped)):
            cells = [
                (wrapped[index][line_index] if line_index < len(wrapped[index]) else '').ljust(column_width)
                for index in range(column_count)
            ]
            self.write(separator.join(cells).rstrip(), size=7, bold=bold, family='DejaVu Sans Mono')

    def draw_line_chart(self, chart, title):
        self.heading(title)
        datasets = chart.get('datasets', [])
        labels = chart.get('labels', [])
        if not datasets or not labels:
            return

        chart_height = 205
        self.ensure_space(chart_height + 10)
        bottom = self.y - chart_height
        axes = self.figure.add_axes([
            MARGIN / PAGE_WIDTH,
            bottom / PAGE_HEIGHT,
            (PAGE_WIDTH - (MARGIN * 2)) / PAGE_WIDTH,
            chart_height / PAGE_HEIGHT,
        ])
        grays = ['0.12', '0.38', '0.62', '0.78', '0.25', '0.52']
        line_styles = ['-', '--', ':', '-.', '-', '--']
        markers = ['o', 's', '^', 'D', 'v', 'P']
        for index, dataset in enumerate(datasets):
            values = dataset.get('data', [])
            axes.plot(
                labels[:len(values)], values,
                color=grays[index % len(grays)],
                linestyle=line_styles[index % len(line_styles)],
                marker=markers[index % len(markers)],
                linewidth=1.4,
                markersize=3.5,
                label=str(dataset.get('label') or dataset.get('year', 'Series')),
            )
        axes.set_xlabel(str(chart.get('x_title', 'Period')), color='#111111', fontsize=8)
        axes.set_ylabel('Consultations', color='#111111', fontsize=8)
        axes.tick_params(axis='both', colors='#111111', labelsize=7)
        axes.grid(axis='y', color='#d0d0d0', linewidth=0.5)
        for spine in axes.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(0.6)
        if len(labels) > 8:
            axes.tick_params(axis='x', labelrotation=35)
            for label in axes.get_xticklabels():
                label.set_horizontalalignment('right')
        axes.legend(
            loc='upper center', bbox_to_anchor=(0.5, 1.18),
            ncol=min(4, len(datasets)), frameon=False, fontsize=7,
        )
        self.y = bottom - 18

    def draw_bar_chart(self, chart):
        labels = chart.get('labels', [])
        values = chart.get('data', [])
        if not values:
            return
        # Wrap category names to fit between ticks and size the gap for their height.
        chart_height = 210
        chart_labels = labels[:len(values)]
        label_width = max(4, min(18, int(150 / max(1, len(chart_labels)))))
        wrapped_labels = [
            '\n'.join(textwrap.wrap(str(label), width=label_width, break_long_words=True, break_on_hyphens=False))
            for label in chart_labels
        ]
        max_label_lines = max((label.count('\n') + 1 for label in wrapped_labels), default=1)
        label_clearance = max(34, (max_label_lines * 9) + 18)
        self.ensure_space(chart_height + label_clearance)
        bottom = self.y - chart_height
        axes = self.figure.add_axes([
            MARGIN / PAGE_WIDTH,
            bottom / PAGE_HEIGHT,
            (PAGE_WIDTH - (MARGIN * 2)) / PAGE_WIDTH,
            chart_height / PAGE_HEIGHT,
        ])
        axes.bar(range(len(values)), values, color='#777777', edgecolor='#222222', linewidth=0.6)
        axes.set_xticks(range(len(wrapped_labels)))
        axes.set_xticklabels(wrapped_labels, rotation=0, ha='center')
        axes.set_ylabel(str(chart.get('y_title', 'Value')), color='#111111', fontsize=8)
        axes.tick_params(axis='both', colors='#111111', labelsize=6.5, pad=3)
        axes.grid(axis='y', color='#d0d0d0', linewidth=0.5)
        axes.set_axisbelow(True)
        for spine in axes.spines.values():
            spine.set_color('#777777')
            spine.set_linewidth(0.6)
        self.y = bottom - label_clearance

    def finish(self):
        if self.figure is not None:
            self.pdf.savefig(self.figure, facecolor='white')
            self.figure.clear()


def _cover(report, writer, logo_path, prepared_by):
    if logo_path:
        try:
            with Image.open(logo_path) as logo:
                axes = writer.figure.add_axes([0.08, 0.905, 0.075, 0.06])
                axes.imshow(logo.convert('L'), cmap='gray')
                axes.axis('off')
        except (OSError, ValueError):
            pass

    writer.figure.text(0.17, 0.95, 'Accudetek Health Diagnostics', fontsize=12,
                       fontweight='bold', color='#111111', va='top')
    writer.figure.text(0.17, 0.925, str(report.get('title', 'Report')), fontsize=10,
                       color='#111111', va='top')
    writer.y = PAGE_HEIGHT - 100
    writer.write(report.get('description', ''), size=9, gap=8)

    sections = report.get('sections', [])
    section_values = {section.get('metric'): str(section.get('value', '')) for section in sections}
    forecast_period = section_values.get('Predicted Month')
    latest_quarter = section_values.get('Latest Quarter Total', '').split(':')[0]
    validation_period = section_values.get('Validation Period')
    metadata = []
    if forecast_period:
        metadata.append(('Forecast period', forecast_period))
    if validation_period:
        metadata.append(('Validation period', validation_period))
    if latest_quarter:
        metadata.append(('Latest quarter represented', latest_quarter))
    covered_period = section_values.get('Data Period Covered') or report.get('historical_period_label')
    if covered_period:
        metadata.append(('Historical data covered', covered_period))
    if report.get('scope_label'):
        scope_label = report['scope_label']
        if report.get('date_range_label'):
            scope_label += f": {report['date_range_label']}"
        metadata.append(('Applied scope', scope_label))
    branch = report.get('branch_name', '')
    if branch:
        branch_label = branch
        if report.get('branch_code'):
            branch_label += f" ({report['branch_code']})"
        metadata.append(('Branch', branch_label))
    if report.get('generated_at'):
        metadata.append(('Generated', report['generated_at']))
    if prepared_by:
        metadata.append(('Prepared by', prepared_by))
        metadata.append(('Signature', '____________________________'))
    for label, value in metadata:
        writer.write(f'{label}: {value}', size=8.5)
    writer.y -= 6
    if report.get('filter_note'):
        writer.write(report['filter_note'], size=8, gap=5)


def _write_summary(writer, sections):
    writer.heading('Report Summary')
    writer.write_table_row(['Metric', 'Value'], bold=True)
    writer.rule()
    for section in sections:
        writer.write_table_row([section.get('metric', ''), section.get('value', '')], bold=True)
        if section.get('explanation'):
            writer.write(section['explanation'], size=8.3, indent=12, gap=3)
        writer.rule()


def _write_main_details(report, writer):
    title = report.get('details_title', 'Details')
    note = report.get('details_note', '')
    chart = report.get('chart_rows') or {}
    if report.get('key') in {'monthly-consultation', 'quarterly'}:
        if chart.get('datasets'):
            writer.draw_line_chart(chart, title)
            x_title = str(chart.get('x_title', 'the displayed periods')).lower()
            writer.write(
                f'Each line shows consultation counts across {x_title}. Compare line heights to compare volumes and follow the line to see how counts change between periods.',
                size=8.5,
            )
            labels = chart.get('labels', [])
            for dataset in chart.get('datasets', []):
                values = dataset.get('data', [])
                if values and labels:
                    peak_index = max(range(min(len(values), len(labels))), key=values.__getitem__)
                    series_label = dataset.get('label') or dataset.get('year', '')
                    writer.write(
                        f"{series_label}: highest displayed count is {values[peak_index]} in {labels[peak_index]}.",
                        size=8.5, indent=12,
                    )
        else:
            writer.heading(title)
            writer.write(note, size=8.5)
            writer.write('No graph data is available yet. Upload consultation records to generate this chart.', size=8.5)
        return

    writer.table(title, note, report.get('rows', []))


def _write_extra_charts(charts, writer):
    for chart in charts:
        writer.heading(chart.get('title', 'Chart'))
        writer.draw_bar_chart(chart)
        y_title = str(chart.get('y_title', 'the displayed measure')).lower()
        writer.write(f'Bars compare {y_title} across categories; taller bars represent higher values.', size=8.5)
        values = chart.get('data', [])
        labels = chart.get('labels', [])
        if values and labels:
            peak_index = max(range(min(len(values), len(labels))), key=values.__getitem__)
            writer.write(f'The highest displayed value is {values[peak_index]} for {labels[peak_index]}.', size=8.5)
        writer.write('Values are shown for each category listed in the chart.', size=8, gap=6)


def _write_extra_tables(tables, writer):
    for table in tables:
        writer.heading(table.get('title', 'Additional Details'))
        writer.write('Forecast values are grouped using the fields shown in the table.', size=8.5)
        rows = table.get('rows', [])
        if rows:
            writer.table('', '', rows)
        else:
            writer.write('No demographic rows are available for this report yet.', size=8.5)


def build_report_pdf(report, logo_path=None, prepared_by=None):
    output = BytesIO()
    with PdfPages(output) as pdf:
        metadata = pdf.infodict()
        metadata['Title'] = str(report.get('title', 'Report'))
        metadata['Author'] = 'Accudetek Health Diagnostics'
        writer = _ReportPdf(pdf)
        _cover(report, writer, logo_path, prepared_by)
        _write_summary(writer, report.get('sections', []))
        _write_main_details(report, writer)
        _write_extra_charts(report.get('extra_charts', []), writer)
        _write_extra_tables(report.get('extra_tables', []), writer)
        writer.finish()
    return output.getvalue()
