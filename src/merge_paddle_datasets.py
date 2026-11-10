"""Merge reviewed Paddle recognition datasets while preserving page splits."""
import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED


def digest(data):
    return hashlib.sha256(data).hexdigest()


def merge(inputs, output, kind, dictionary, corrections=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    alphabet = set(Path(dictionary).read_text(encoding='utf-8-sig').splitlines()) | {' '}
    dict_hash = digest(Path(dictionary).read_bytes())
    rows, page_splits, hash_labels, sources = [], {}, {}, []
    fixes = json.loads(Path(corrections).read_text(encoding='utf-8')) if corrections else {}
    used_fixes = set()
    for number, path in enumerate(map(Path, inputs)):
        archive_bytes = path.read_bytes()
        with ZipFile(path) as z:
            problem = z.testzip()
            if problem:
                raise ValueError(f'Corrupt archive entry: {problem}')
            manifests = [name for name in z.namelist() if PurePosixPath(name).name == 'manifest.json']
            if len(manifests) != 1:
                raise ValueError(f'Expected one manifest: {path}')
            prefix = manifests[0][:-len('manifest.json')]
            manifest = json.loads(z.read(manifests[0]))
            sources.append({'archive': str(path.resolve()), 'sha256': digest(archive_bytes)})
            by_image = {}
            for item in manifest['cells']:
                name = item['image']
                if name in by_image:
                    raise ValueError(f'Duplicate image in manifest: {name}')
                by_image[name] = item
            seen = set()
            for split in ('train', 'val'):
                labels = z.read(prefix + split + '.txt').decode('utf-8-sig').splitlines()
                for line in labels:
                    name, text = line.split('\t', 1)
                    p = PurePosixPath(name)
                    if p.is_absolute() or '..' in p.parts or name in seen or name not in by_image:
                        raise ValueError(f'Unsafe or repeated image: {name}')
                    seen.add(name)
                    item = by_image[name]
                    if item['split'] != split or item['text'] != text or item['review_status'] != 'reviewed':
                        raise ValueError(f'Manifest mismatch: {name}')
                    original_text = text
                    if name in fixes:
                        used_fixes.add(name)
                        if fixes[name] is None:
                            continue
                        text = fixes[name]
                    if not text.strip() or any(x in text for x in '\t\r\n\0'):
                        raise ValueError(f'Invalid label: {name}')
                    if set(text) - alphabet:
                        raise ValueError(f'Characters outside dictionary: {name}: {set(text) - alphabet}')
                    if kind == 'value' and ',' in text or kind == 'headers' and any(x in text for x in '‐‑‒–—−'):
                        raise ValueError(f'Unnormalized label: {name}')
                    raw = z.read(prefix + name)
                    h = digest(raw)
                    if h != item['output_sha256']:
                        raise ValueError(f'Image hash mismatch: {name}')
                    page = (item['book'], item['page_name'])
                    if page in page_splits and page_splits[page] != split:
                        raise ValueError(f'Page appears in both splits: {page}')
                    page_splits[page] = split
                    if h in hash_labels:
                        prev_split, prev_text = hash_labels[h]
                        if (prev_split, prev_text) != (split, text):
                            raise ValueError(f'Duplicate image crosses splits or has different label: {name}')
                    hash_labels[h] = (split, text)
                    record = dict(item)
                    record.update(image=f'images/source{number:02d}_{p.name}', text=text,
                                  transcription=text, source_group=item.get('source_group', 'other'),
                                  output_sha256=h, split=split)
                    if text != original_text:
                        record.update(previous_text=original_text, correction='visually reviewed script correction')
                    rows.append((record, raw))
            if len(seen) != len(by_image):
                raise ValueError(f'Entries missing in split files: {path}')
    if used_fixes != set(fixes):
        raise ValueError(f'Unused corrections: {set(fixes) - used_fixes}')
    if not all(any(r['split'] == s for r, _ in rows) for s in ('train', 'val')):
        raise ValueError('Both train and val must be nonempty')
    names = [r['image'] for r, _ in rows]
    if len(names) != len(set(names)):
        raise ValueError('Output image name collision')
    summary = {}
    for split in ('train', 'val'):
        group = [r for r, _ in rows if r['split'] == split]
        summary[split] = {'cells': len(group), 'pages': len({(r['book'], r['page_name']) for r in group}),
                          'groups': dict(Counter(r['source_group'] for r in group))}
    manifest = {'format': f'paddleocr_recognition_{kind}_v1', 'dictionary_sha256': dict_hash,
                'split_unit': 'source_page_preserved', 'confirmed_pending_ids': [],
                'padding_mode': 'preserve_each_source_crop', 'sources': sources,
                'summary': summary, 'cells': [r for r, _ in rows]}
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.parent / ('.' + output.name + '.' + uuid4().hex + '.tmp')
    try:
        with ZipFile(temp, 'w', ZIP_DEFLATED, compresslevel=5) as z:
            for record, raw in rows:
                z.writestr(record['image'], raw)
            for split in ('train', 'val'):
                z.writestr(split + '.txt', ''.join(f"{r['image']}\t{r['text']}\n" for r, _ in rows if r['split'] == split))
            z.writestr('manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
        with ZipFile(temp) as z:
            if z.testzip():
                raise ValueError('Output ZIP failed CRC check')
            if len(json.loads(z.read('manifest.json'))['cells']) != len(rows):
                raise ValueError('Output manifest count mismatch')
        os.rename(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    return {'output': str(output), 'sha256': digest(output.read_bytes()), 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=['value', 'headers'], required=True)
    parser.add_argument('--dictionary', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--corrections', help='JSON mapping source image names to visually verified text; null excludes')
    parser.add_argument('inputs', nargs='+')
    args = parser.parse_args()
    print(json.dumps(merge(args.inputs, args.output, args.kind, args.dictionary, args.corrections), ensure_ascii=False, indent=2))
