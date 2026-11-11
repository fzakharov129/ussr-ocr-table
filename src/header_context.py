from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
from pathlib import Path
import re

from .entities import CellType, RECOGNIZED
from .normalization import _dictionary_key, load_header_dictionary


def _dictionary():
    dictionary = load_header_dictionary(Path(__file__).resolve().parent.parent / 'header_dictionary.yaml')
    regions = {}
    for republic, names in dictionary['republic_regions'].items():
        regions[republic] = republic
        regions.update({name: republic for name in names})
    canonical = set(regions) | set(dictionary['aliases'].values()) | set(dictionary.get('service_headers', []))
    return canonical, regions


def _refresh(data):
    for page, table in data['pages'].items():
        issues = []
        groups = defaultdict(list)
        for axis in ('rows', 'columns'):
            for item in table[axis]:
                if item['needs_review'] or not item['name']:
                    issues.append(dict(page_name=page, cell_id=item['cell_id'], row_name=None,
                                       column_name=None, value=item['name'],
                                       reason='Заголовок или контекст требует проверки'))
        for record in table['records']:
            record['needs_review'] = (
                record['cell_type'] == CellType.VALUE
                and record['recognition_status'] != RECOGNIZED
            )
            r, c = record['row_index'], record['column_index']
            if r is None or c is None:
                record['needs_review'] = True
                issues.append(dict(page_name=page, cell_id=record['cell_id'], row_name=None,
                                   column_name=None, value=record['value'],
                                   reason='Нет единственного заголовка строки или столбца'))
                continue
            row, column = table['rows'][r], table['columns'][c]
            record.update(row_name=row['name'], row_republic=row['republic'],
                          column_name=column['name'], column_republic=column['republic'])
            if row['needs_review'] or column['needs_review'] or not row['name'] or not column['name']:
                record['needs_review'] = True
                issues.append(dict(page_name=page, cell_id=record['cell_id'], row_name=row['name'],
                                   column_name=column['name'], value=record['value'],
                                   reason='Заголовок или контекст республики требует проверки'))
            if record['cell_type'] == CellType.VALUE and type(record['value']) not in (int, float):
                record['needs_review'] = True
                issues.append(dict(page_name=page, cell_id=record['cell_id'], row_name=row['name'],
                                   column_name=column['name'], value=record['value'],
                                   reason='В VALUE нет числового значения'))
            groups[(r, c)].append(record)
        table['issues'] = issues
        table['values'] = [[None for _ in table['columns']] for _ in table['rows']]
        table['positions'] = []
        for (r, c), records in groups.items():
            distinct = list(dict.fromkeys(record['value'] for record in records))
            conflict = len(distinct) > 1
            if not conflict:
                table['values'][r][c] = distinct[0]
            table['positions'].append(dict(
                row_index=r, column_index=c, conflict=conflict,
                needs_review=conflict or len(records) > 1 or any(x['needs_review'] for x in records),
                sources=[dict(page_name=page, cell_id=x['cell_id']) for x in records],
            ))
    return data


def _trusted(item, canonical):
    return bool(item['name']) and not item['needs_review'] and item['name'] in canonical


def _observations(pages, regions):
    observations = defaultdict(list)
    connections = defaultdict(set)
    for page, table in pages.items():
        for axis in ('rows', 'columns'):
            items = table[axis]
            counts = Counter(item['name'] for item in items)
            for item in items:
                if (item['name'] in regions and not item['needs_review']
                        and counts[item['name']] == 1):
                    observations[item['name']].append(dict(
                        name=item['name'], axis=axis, evidence_page=page,
                        evidence_cell_id=item['cell_id'],
                    ))
            if axis != 'rows':
                continue
            for left, right in zip(items, items[1:]):
                if (left['name'] in regions and right['name'] in regions
                        and not left['needs_review'] and not right['needs_review']
                        and counts[left['name']] == counts[right['name']] == 1
                        and left['name'] != right['name']):
                    connections[(left['name'], right['name'])].add(page)
    return observations, connections


def _rank(variants):
    ranked = []
    for name, evidence in variants.items():
        rows = {item['evidence_page'] for item in evidence if item['axis'] == 'rows'}
        columns = {item['evidence_page'] for item in evidence if item['axis'] == 'columns'}
        ranked.append((2 * len(rows) + len(columns), name, evidence))
    return sorted(ranked, key=lambda item: (-item[0], item[1]))


def _reference_sequences(observations, connections):
    edges = {
        pair: pages for pair, pages in connections.items()
        if len(pages) >= 2 and len(connections.get((pair[1], pair[0]), set())) < 2
    }
    neighbors = {name: set() for name in observations}
    for first, second in edges:
        neighbors[first].add(second)
        neighbors[second].add(first)
    references = []
    unseen = set(neighbors)
    while unseen:
        start = min(unseen)
        pending = [start]
        component = set()
        while pending:
            name = pending.pop()
            if name in component:
                continue
            component.add(name)
            pending.extend(neighbors[name] - component)
        unseen.difference_update(component)
        outgoing = {name: set() for name in component}
        indegree = {name: 0 for name in component}
        for first, second in edges:
            if first in component and second in component:
                outgoing[first].add(second)
                indegree[second] += 1
        available = sorted(name for name, degree in indegree.items() if degree == 0)
        ordered = []
        ambiguous = False
        while available:
            if len(available) > 1:
                ambiguous = True
            name = available.pop(0)
            ordered.append(name)
            for target in sorted(outgoing[name]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    available.append(target)
            available.sort()
        if len(ordered) != len(component):
            ambiguous = True
            ordered.extend(sorted(component - set(ordered)))
        positions = []
        pages = set()
        for index, name in enumerate(ordered):
            evidence = observations[name]
            pages.update(item['evidence_page'] for item in evidence)
            ranked = _rank({name: evidence})
            positions.append(dict(index=index, name=name, candidates=[dict(
                name=name, weight=ranked[0][0], evidence=evidence,
            )]))
        links = [dict(first=first, second=second, pages=sorted(sources))
                 for (first, second), sources in sorted(edges.items())
                 if first in component and second in component]
        references.append(dict(pages=sorted(pages), positions=positions,
                               connections=links, order_uncertain=ambiguous))
    return references


def _name_key(name):
    name = re.sub(r'\bобл\.?\s*$', 'область', name, flags=re.IGNORECASE)
    return _dictionary_key(name)


def _candidates(item, observations, regions):
    hierarchy, kind, stem = _name_key(item['name'])
    if not stem:
        return []
    candidates = []
    for name, evidence in observations.items():
        target_hierarchy, target_kind, target_stem = _name_key(name)
        if hierarchy != target_hierarchy or (kind and target_kind and kind != target_kind):
            continue
        if len({entry['evidence_page'] for entry in evidence}) < 2:
            continue
        if not any(entry['axis'] == 'rows' for entry in evidence):
            continue
        similarity = SequenceMatcher(None, stem, target_stem).ratio()
        if similarity >= 0.90:
            candidates.append(dict(name=name, republic=regions.get(name),
                                   similarity=similarity, evidence=evidence))
    return sorted(candidates, key=lambda item: (-item['similarity'], item['name']))


def review_sequences(data):
    data = deepcopy(data)
    _, regions = _dictionary()
    pages = data['pages']
    observations, connections = _observations(pages, regions)
    references = _reference_sequences(observations, connections)
    corrections, suggestions = [], []
    for page, table in pages.items():
        for axis in ('rows', 'columns'):
            for item in table[axis]:
                original = item['name']
                if not original or original in regions or original == 'Всего по республике':
                    continue
                candidates = _candidates(item, observations, regions)
                if not candidates:
                    continue
                best = candidates[0]
                unique = len(candidates) == 1 or best['similarity'] - candidates[1]['similarity'] >= 0.08
                independent = [entry for entry in best['evidence']
                               if not (entry['evidence_page'] == page
                                       and entry['axis'] == axis
                                       and entry['evidence_cell_id'] == item['cell_id'])]
                if unique and independent:
                    item.setdefault('original_name', original)
                    item['name'] = best['name']
                    item['republic'] = best['republic']
                    item['needs_review'] = False
                    corrections.append(dict(page_name=page, axis=axis, cell_id=item['cell_id'],
                                            original=original, corrected=best['name'],
                                            evidence=independent))
                else:
                    item['needs_review'] = True
                    evidence = [entry for candidate in candidates for entry in candidate['evidence']]
                    suggestions.append(dict(page_name=page, axis=axis, cell_id=item['cell_id'],
                                            original=original, candidates=evidence))
    data['reference_sequences'] = references
    data['corrections'] = corrections
    data['suggestions'] = suggestions
    return _refresh(data)


def resolve_republic_headers(data):
    data = deepcopy(data)
    contexts = []
    for page, table in data['pages'].items():
        columns = table['columns']
        for index, item in enumerate(columns):
            if item['name'] != 'Всего по республике':
                continue
            group = []
            for neighbor in columns[index + 1:]:
                if neighbor['name'] == 'Всего по республике':
                    break
                group.append(neighbor)
            republic = None
            accepted = []
            disputed = []
            boundary = len(group)
            if group and group[0]['republic'] and not group[0]['needs_review']:
                candidate = group[0]['republic']
                position = 0
                while position < len(group):
                    neighbor = group[position]
                    if neighbor['republic'] == candidate and not neighbor['needs_review']:
                        accepted.append(neighbor)
                        position += 1
                        continue
                    if (position + 1 < len(group)
                            and group[position + 1]['republic'] == candidate
                            and not group[position + 1]['needs_review']):
                        disputed.append(neighbor)
                        position += 1
                        continue
                    boundary = position
                    break
                resumes = any(
                    neighbor['republic'] == candidate and not neighbor['needs_review']
                    for neighbor in group[boundary:]
                )
                if len(accepted) >= 2 and not resumes:
                    republic = candidate
                    for neighbor in disputed:
                        neighbor['needs_review'] = True
            evidence = [dict(cell_id=x['cell_id'], name=x['name'], republic=x['republic'],
                             needs_review=x['needs_review']) for x in group]
            item['republic'] = republic
            item['needs_review'] = (republic is None or
                                    data['graphs'][page].nodes[item['cell_id']].recognition_status != RECOGNIZED)
            suffix = republic or 'требует проверки'
            item['display_name'] = f"{item['name']} — {suffix}"
            item['context_cell_ids'] = [x['cell_id'] for x in accepted]
            contexts.append(dict(page_name=page, cell_id=item['cell_id'], republic=republic,
                                 evidence=evidence,
                                 disputed_cell_ids=[x['cell_id'] for x in disputed] if republic else []))
    data['republic_contexts'] = contexts
    return _refresh(data)
