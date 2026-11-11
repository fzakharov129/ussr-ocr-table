from html import escape
from pathlib import Path

from IPython.display import HTML, display

from .storage import BookStorage, book_output_dir, source_fingerprint
from .validations import PipelineChecks


def show_page_table(book, page_index=0, *, book_id=None, outputs_dir=None, show_ids=True,
                    stage="republic_headers"):
    book = Path(book).resolve()
    folder = book_output_dir(book, outputs_dir, book_id)
    if stage not in ('page_tables', 'sequence_review', 'republic_headers'):
        raise ValueError('Неверный этап таблиц страниц')
    data = BookStorage(folder, source_fingerprint(book)).load(stage)
    PipelineChecks.state(stage, data)
    names = list(data['pages'])
    if not isinstance(page_index, int) or isinstance(page_index, bool) or not 0 <= page_index < len(names):
        raise ValueError(f'Номер страницы должен быть от 0 до {len(names) - 1}')
    page = names[page_index]
    table = data['pages'][page]
    def label(item):
        name = item.get('display_name') or item['name'] or '(нет текста)'
        if not item.get('display_name') and item['republic'] and name != item['republic']:
            name += ' — ' + item['republic']
        if show_ids:
            name += f" [ID {item['cell_id']}]"
        if item['needs_review']:
            name += ' ⚠'
        return escape(name)
    positions = {(p['row_index'], p['column_index']): p for p in table['positions']}
    lines = [f'<h3>{escape(page)} ({page_index + 1}/{len(names)})</h3>',
             '<div style="overflow:auto"><table><thead><tr><th>Строка / столбец</th>']
    lines.extend(f'<th>{label(col)}</th>' for col in table['columns'])
    lines.append('</tr></thead><tbody>')
    for i, row in enumerate(table['rows']):
        lines.append(f'<tr><th>{label(row)}</th>')
        for j, value in enumerate(table['values'][i]):
            position = positions.get((i, j))
            text = 'КОНФЛИКТ' if position and position['conflict'] else ('' if value is None else str(value))
            style = 'background:#ffcccc' if position and position['conflict'] else (
                'background:#fff2b3' if position and position['needs_review'] else '')
            lines.append(f'<td style="{style}">{escape(text)}</td>')
        lines.append('</tr>')
    lines.append('</tbody></table></div>')
    display(HTML(''.join(lines)))
    print(f"Замечаний: {len(table['issues'])}; возможных совпадений: "
          f"{sum(s['page_name'] == page for s in data['suggestions'])}")
    return table


def show_page_matches(book, page_index=0, *, book_id=None, outputs_dir=None):
    book = Path(book).resolve()
    folder = book_output_dir(book, outputs_dir, book_id)
    data = BookStorage(folder, source_fingerprint(book)).load('sequence_review')
    PipelineChecks.state('sequence_review', data)
    names = list(data['pages'])
    if not isinstance(page_index, int) or isinstance(page_index, bool) or not 0 <= page_index < len(names):
        raise ValueError(f'Номер страницы должен быть от 0 до {len(names) - 1}')
    page = names[page_index]
    matches = [s for s in data['suggestions'] if s['page_name'] == page]
    corrections = [item for item in data.get('corrections', []) if item['page_name'] == page]
    for item in corrections:
        print(f"Исправлено [{item['cell_id']}]: {item['original']} → {item['corrected']}")
        for evidence in item['evidence']:
            print(f"  {evidence['evidence_page']} [{evidence['evidence_cell_id']}]")
    for item in matches:
        print(f"На проверку [{item['cell_id']}]: {item['original']}")
        for candidate in item['candidates']:
            print(f"  {candidate['name']}: {candidate['evidence_page']} [{candidate['evidence_cell_id']}]")
    print(f'Исправлений: {len(corrections)}; неоднозначных совпадений: {len(matches)}')
    return {'corrections': corrections, 'suggestions': matches}


def show_reference_sequences(book, *, book_id=None, outputs_dir=None):
    book = Path(book).resolve()
    folder = book_output_dir(book, outputs_dir, book_id)
    data = BookStorage(folder, source_fingerprint(book)).load("sequence_review")
    PipelineChecks.state("sequence_review", data)
    references = data.get("reference_sequences", [])
    for index, reference in enumerate(references, start=1):
        print(f"Общая последовательность {index}")
        print(f"Страниц-источников: {len(reference['pages'])}")
        if reference.get('order_uncertain'):
            print("Порядок содержит неоднозначности и требует проверки")
        for position in reference['positions']:
            name = position['name'] or "требует проверки"
            print(f"  {position['index'] + 1}. {name}")
            for candidate in position['candidates']:
                sources = ", ".join(
                    f"{item['axis']}: {item['evidence_page']} [{item['evidence_cell_id']}]"
                    for item in candidate['evidence']
                )
                print(f"     {candidate['name']} (вес {candidate['weight']}): {sources}")
    print(f"Опорных последовательностей: {len(references)}")
    return references
