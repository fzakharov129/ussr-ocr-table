from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
import re
from tempfile import TemporaryDirectory

from .entities import CellDirection, CellType, RECOGNIZED
from .export import _header_text, _issue, _nearest_headers
from .normalization import load_header_dictionary
from .validations import ExportChecks


def _headers(graph, label):
    axis = (lambda c: (c.center[1], c.center[0], c.cell_id)) if label == CellType.ROW_HEADER else (
        lambda c: (c.center[0], c.center[1], c.cell_id)
    )
    return sorted((c for c in graph.nodes.values() if c.cell_label == label), key=axis)


def _page_table(graph, canonical):
    rows = _headers(graph, CellType.ROW_HEADER)
    columns = _headers(graph, CellType.COLUMN_HEADER)
    row_items = [dict(page_name=graph.page_name, cell_id=c.cell_id, name=_header_text(c),
                      republic=c.republic, needs_review=c.recognition_status != RECOGNIZED or _header_text(c) not in canonical)
                 for c in rows]
    col_items = [dict(page_name=graph.page_name, cell_id=c.cell_id, name=_header_text(c),
                      republic=c.republic, needs_review=c.recognition_status != RECOGNIZED or _header_text(c) not in canonical,
                      context_cell_ids=[]) for c in columns]
    item_by_id = {item['cell_id']: item for item in row_items + col_items}
    row_index = {c.cell_id: i for i, c in enumerate(rows)}
    col_index = {c.cell_id: i for i, c in enumerate(columns)}
    records, issues, groups = [], [], defaultdict(list)
    for cell in sorted(graph.nodes.values(), key=lambda c: (c.center[1], c.center[0], c.cell_id)):
        if cell.cell_label in {CellType.ROW_HEADER, CellType.COLUMN_HEADER}:
            if item_by_id[cell.cell_id]['needs_review'] or not _header_text(cell):
                issues.append(_issue(dict(page_name=graph.page_name, cell_id=cell.cell_id,
                                          value=cell.recognized_value), "Заголовок требует проверки"))
            continue
        if cell.cell_label != CellType.VALUE:
            continue
        row_ids = _nearest_headers(graph, cell.cell_id, CellDirection.LEFT, CellType.ROW_HEADER)
        col_ids = _nearest_headers(graph, cell.cell_id, CellDirection.UP, CellType.COLUMN_HEADER)
        record = dict(page_name=graph.page_name, cell_id=cell.cell_id, row_header_ids=row_ids,
                      column_header_ids=col_ids, value=cell.recognized_value if cell.cell_label == CellType.VALUE else None,
                      cell_type=cell.cell_label, recognition_status=cell.recognition_status,
                      row_index=None, column_index=None,
                      needs_review=cell.cell_label == CellType.VALUE and cell.recognition_status != RECOGNIZED)
        records.append(record)
        if len(row_ids) != 1 or len(col_ids) != 1:
            record['needs_review'] = True
            issues.append(_issue(record, 'Нет единственного заголовка строки или столбца'))
            continue
        row, col = row_items[row_index[row_ids[0]]], col_items[col_index[col_ids[0]]]
        record.update(row_index=row_index[row_ids[0]], column_index=col_index[col_ids[0]],
                      row_name=row['name'], row_republic=row['republic'],
                      column_name=col['name'], column_republic=col['republic'])
        if not row['name'] or not col['name'] or row['needs_review'] or col['needs_review']:
            record['needs_review'] = True
            issues.append(_issue(record, 'Заголовок или контекст республики требует проверки'))
        if cell.cell_label == CellType.VALUE and type(cell.recognized_value) not in (int, float):
            record['needs_review'] = True
            issues.append(_issue(record, 'В VALUE нет числового значения'))
        groups[(record['row_index'], record['column_index'])].append(record)
    values = [[None for _ in col_items] for _ in row_items]
    positions = []
    for (r, c), sources in groups.items():
        distinct = list(dict.fromkeys(item['value'] for item in sources))
        conflict = len(distinct) > 1
        if not conflict:
            values[r][c] = distinct[0]
        positions.append(dict(row_index=r, column_index=c, conflict=conflict,
                              needs_review=conflict or len(sources) > 1 or any(s['needs_review'] for s in sources),
                              sources=[dict(page_name=s['page_name'], cell_id=s['cell_id']) for s in sources]))
    return dict(rows=row_items, columns=col_items, values=values, positions=positions,
                records=records, issues=issues)


def build_page_tables(graphs):
    ExportChecks.graphs(graphs)
    dictionary = load_header_dictionary(Path(__file__).resolve().parent.parent / 'header_dictionary.yaml')
    canonical = set(dictionary['aliases'].values()) | set(dictionary.get('service_headers', []))
    for republic, regions in dictionary['republic_regions'].items():
        canonical.add(republic)
        canonical.update(regions)
    pages = {name: _page_table(graph, canonical) for name, graph in graphs.items()}
    return dict(graphs=graphs, pages=pages, suggestions=[])


def assemble_page_tables(data):
    pages = data['pages']
    counts = {axis: {page: Counter((x['name'], x['republic']) for x in table[axis])
                     for page, table in pages.items()} for axis in ('rows', 'columns')}
    duplicate_keys = {axis: {key for counter in counts[axis].values() for key, count in counter.items() if count > 1}
                      for axis in ('rows', 'columns')}
    axes = {'rows': [], 'columns': []}
    indices = {'rows': {}, 'columns': {}}
    page_maps = {}
    for page, table in pages.items():
        page_maps[page] = {}
        for axis in ('rows', 'columns'):
            mapping = []
            for item in table[axis]:
                semantic = (item['name'], item['republic'])
                reliable = bool(item['name']) and not item['needs_review'] and semantic not in duplicate_keys[axis]
                if axis == 'columns' and not item['republic']:
                    reliable = False
                key = ('shared', *semantic) if reliable else ('local', page, item['cell_id'])
                if key not in indices[axis]:
                    indices[axis][key] = len(axes[axis])
                    axes[axis].append(dict(name=item['name'], republic=item['republic'],
                                           needs_review=item['needs_review'], sources=[]))
                index = indices[axis][key]
                source = dict(page_name=page, cell_id=item['cell_id'])
                if source not in axes[axis][index]['sources']:
                    axes[axis][index]['sources'].append(source)
                mapping.append(index)
            page_maps[page][axis] = mapping
    records, issues, groups = [], [], defaultdict(list)
    for page, table in pages.items():
        issues.extend(table['issues'])
        for original in table['records']:
            record = dict(original)
            if record['row_index'] is not None and record['column_index'] is not None:
                record['row_index'] = page_maps[page]['rows'][record['row_index']]
                record['column_index'] = page_maps[page]['columns'][record['column_index']]
                groups[(record['row_index'], record['column_index'])].append(record)
            records.append(record)
    for suggestion in data['suggestions']:
        issues.append(dict(page_name=suggestion['page_name'], cell_id=suggestion['cell_id'],
                           row_name=suggestion['original'], column_name=None, value=None,
                           reason='Неоднозначное совпадение заголовка на другой странице'))
    values = [[None for _ in axes['columns']] for _ in axes['rows']]
    positions = []
    for (r, c), sources in groups.items():
        distinct = list(dict.fromkeys(s['value'] for s in sources))
        conflict = len(distinct) > 1
        duplicate = any(n > 1 for n in Counter(s['page_name'] for s in sources).values())
        if not conflict:
            values[r][c] = distinct[0]
        positions.append(dict(row_index=r, column_index=c, conflict=conflict,
                              needs_review=conflict or duplicate or any(s['needs_review'] for s in sources),
                              sources=[dict(page_name=s['page_name'], cell_id=s['cell_id']) for s in sources]))
        for source in sources:
            if conflict:
                issues.append(_issue(source, 'Разные значения для общей позиции'))
            if duplicate:
                issues.append(_issue(source, 'Несколько ячеек одной страницы попали в общую позицию'))
    return dict(rows=axes['rows'], columns=axes['columns'], values=values, positions=positions,
                records=records, issues=issues)


def save_page_tables(pages, output_dir, axis_label="Строка / Столбец"):
    from .export import _write_xlsx

    if not isinstance(pages, dict) or not pages:
        raise ValueError("Нет таблиц страниц для сохранения")
    for table in pages.values():
        ExportChecks.excel(table)
    folder = Path(output_dir) / "pages_xlsx"
    folder.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".pages_", dir=folder) as temporary:
        prepared = Path(temporary) / "pages"
        prepared.mkdir()
        for index, (page_name, table) in enumerate(pages.items(), start=1):
            stem = PurePosixPath(page_name).stem
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", stem).strip(" .")
            if not stem:
                stem = "page"
            path = prepared / f"{index:03d}_{stem}.xlsx"
            _write_xlsx(path, page_name, table, axis_label)
        for path in prepared.iterdir():
            path.replace(folder / path.name)
    return folder
