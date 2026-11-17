"""Load simple Label Studio extraction pilots without aggregating annotations.

Supported controls: a per-region Choices control, an optional task-level
missing Choices control, and configurable TextArea controls (including text
attached to a rectangle). Other relationships and text spans are not supported.
For CORD, ``documents_file`` adds OCR text and identifies original regions.
Text fields map control names to field names; reference columns map field
names to CSV columns. An empty region-reference list is not ground truth for
absence. Annotation timing is always per submission, not per field.
"""

import json
from pathlib import Path
import warnings

from urllib.parse import urlparse, parse_qs
import math
import matplotlib.pyplot as plt
from PIL import Image


import pandas as pd

from .analysis import read_label_studio, flag_timing, add_annotation_sequence


def _references(path, id_column):
    if path is None:
        return None
    frame = pd.read_csv(path, dtype={id_column: 'string'})
    if id_column not in frame:
        raise ValueError(f'Reference file must contain {id_column!r}.')
    ids = frame[id_column]
    if ids.isna().any() or ids.str.strip().eq('').any() or ids.duplicated().any():
        raise ValueError('Reference IDs must be nonempty and unique.')
    return frame.rename(columns={id_column: 'example_id'})


def _region_list(value):
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    try:
        result = json.loads(value) if isinstance(value, str) else value
    except (TypeError, ValueError) as error:
        raise ValueError('Reference region IDs must be a JSON list.') from error
    if not isinstance(result, list) or any(not isinstance(x, str) for x in result):
        raise ValueError('Reference region IDs must be a JSON list of strings.')
    return result


def load_ie_pilot(annotations_file, reference_file=None, *, documents_file=None,
                  id_field='example_id', reference_id_col='example_id',
                  field_name='total_amount', region_control='is_total',
                  missing_control='missing_total', region_choice='Final total',
                  missing_choice='No OCR box contains the total',
                  reference_region_col='reference_total_line_ids',
                  text_fields=None, reference_value_cols=None):
    """Return items, annotations and field responses for one pilot export.

    ``text_fields`` maps TextArea control names to field names; e.g.
    ``{'person': 'name', 'date_input': 'date'}``. Reference CSV columns are
    specified with ``reference_value_cols={'name': 'reference_name'}``.
    Set ``region_control=None`` for a text-only task. Unknown result types or
    controls raise an error instead of being interpreted as classifications.
    Preloaded ``ocr_box`` rectangles are context, never responses. Cancelled
    records remain visible but are excluded from submission coverage.
    """
    text_fields = text_fields or {}
    reference_value_cols = reference_value_cols or {}
    tasks = read_label_studio(annotations_file)
    reference = _references(reference_file, reference_id_col)
    if reference is not None:
        for column in ([reference_region_col] if region_control and reference_region_col else []) + list(reference_value_cols.values()):
            if column not in reference:
                raise ValueError(f'Reference file has no column {column!r}.')
    documents = {}
    if documents_file is not None:
        with Path(documents_file).open(encoding='utf-8') as stream:
            documents = {doc['document_id']: doc for doc in map(json.loads, stream)}
    references = {} if reference is None else reference.set_index('example_id').to_dict('index')
    task_rows, annotation_rows, response_rows = [], [], []
    seen = set()
    expected_fields = ([field_name] if region_control else []) + list(dict.fromkeys(text_fields.values()))
    if not expected_fields:
        raise ValueError('Configure a region control or at least one text field.')
    for task in tasks:
        data = task.get('data') or {}
        example = data.get(id_field)
        if example is None or not str(example).strip():
            raise ValueError(f'Task {task.get("id")!r} is missing {id_field!r}.')
        example = str(example)
        if example in seen:
            raise ValueError(f'Duplicate exported task for {example!r}; cannot identify one source document.')
        seen.add(example)
        doc = documents.get(example, {})
        ocr = {line['line_id']: line.get('text', '') for line in doc.get('lines', [])}
        # Predictions identify the candidate universe even without documents.jsonl.
        prediction_ids = {r.get('id') for p in task.get('predictions') or []
                          if isinstance(p, dict) for r in p.get('result') or [] if isinstance(r, dict) and r.get('from_name') == 'ocr_box' and r.get('id')}
        known_ids = set(ocr) | prediction_ids
        options = data.get('options') or []
        if isinstance(options, list):
            for option in options:
                if isinstance(option, dict) and option.get('alias') and option.get('alias') != 'no_total':
                    known_ids.add(option['alias'])
                    ocr.setdefault(option['alias'], str(option.get('value', '')).partition(' | ')[2])
        task_rows.append({'example_id': example, 'task_id': task.get('id'),
                          'text': data.get('text'), 'image': data.get('image'),
                          'ocr_line_count': len(known_ids) if known_ids else None,
                          'task_exported': True})
        records = task.get('annotations') or []
        if not isinstance(records, list):
            raise ValueError('Task annotations must be a list.')
        for annotation in records:
            if not isinstance(annotation, dict):
                raise ValueError('Each annotation must be an object.')
            results = annotation.get('result') or []
            if not isinstance(results, list) or any(not isinstance(r, dict) for r in results):
                raise ValueError('Annotation results must be a list of objects.')
            person = annotation.get('completed_by')
            if isinstance(person, dict):
                person = person.get('id')
            cancelled = bool(annotation.get('was_cancelled', False))
            base = {'example_id': example, 'annotation_id': annotation.get('id'), 'annotator_id': person}
            selected, missing, texts = [], False, {field: [] for field in text_fields.values()}
            geometry = {r.get('id'): r for r in results if r.get('from_name') == 'ocr_box'}
            for result in results:
                control, kind = result.get('from_name'), result.get('type')
                value = result.get('value') or {}
                if not isinstance(value, dict):
                    raise ValueError(f'Invalid result value for {control!r}.')
                if control == 'ocr_box' and kind in ('rectangle', 'rectanglelabels'):
                    continue
                if region_control and control == region_control and kind == 'choices':
                    choices = value.get('choices', [])
                    if not isinstance(choices, list):
                        raise ValueError('Choices must be a list.')
                    if region_choice in choices:
                        selected.append(result)
                    elif choices:
                        raise ValueError(f'Unexpected choice for {region_control!r}: {choices!r}')
                elif missing_control and control == missing_control and kind == 'choices':
                    choices = value.get('choices', [])
                    if not isinstance(choices, list) or any(x != missing_choice for x in choices):
                        raise ValueError(f'Unexpected choice for {missing_control!r}.')
                    missing = missing or bool(choices)
                elif control in text_fields and kind == 'textarea':
                    values = value.get('text', [])
                    if not isinstance(values, list):
                        raise ValueError('TextArea values must be a list.')
                    for entered in values:
                        if isinstance(entered, str) and entered.strip():
                            texts[text_fields[control]].append((entered, result))
                elif kind == 'rectanglelabels' and any(r.get('id') == result.get('id') and r.get('from_name') in text_fields for r in results):
                    # A drawn rectangle supporting a per-region TextArea.
                    geometry[result.get('id')] = result
                else:
                    raise ValueError(f'Unsupported result control/type: {control!r}/{kind!r}. Configure a supported extraction control.')
            fields = []
            if region_control:
                ref = references.get(example, {})
                refs = _region_list(ref.get(reference_region_col)) if reference_region_col else []
                outcome = 'ambiguous' if len(selected) > 1 or (selected and missing) else 'selected' if selected else 'missing' if missing else 'unanswered'
                for result in selected or [None]:
                    rid = result.get('id') if result else None
                    box = geometry.get(rid, result or {})
                    box_value = box.get('value', {})
                    known = (rid in known_ids) if rid and known_ids else None
                    if outcome == 'selected' and not rid:
                        outcome = 'ambiguous'
                    match = (rid in refs) if outcome == 'selected' and refs and known is not False else pd.NA
                    comparison = ('ambiguous' if outcome == 'ambiguous' else 'unanswered' if outcome == 'unanswered' else
                                  'no reference' if not refs else 'reported missing' if outcome == 'missing' else
                                  'manual region' if known is False else 'match' if match is True else 'different')
                    fields.append({**base, 'field_name': field_name, 'value': ocr.get(rid), 'region_id': rid,
                                   'reference_region_ids': refs, 'reference_value': None,
                                   'reference_match': match, 'comparison': comparison, 'status': outcome,
                                   'region_source': 'ocr' if known is True else 'manual' if known is False else 'unknown',
                                   'geometry': {k: box_value[k] for k in ('x','y','width','height','rotation') if k in box_value} or None})
            for field, values in texts.items():
                ref = references.get(example, {}).get(reference_value_cols.get(field))
                ref = None if ref is None or pd.isna(ref) else str(ref)
                status = 'ambiguous' if len(values) > 1 else 'selected' if values else 'unanswered'
                for entered, result in values or [(None, {})]:
                    match = entered == ref if status == 'selected' and ref is not None else pd.NA
                    rid = result.get('id'); box = geometry.get(rid, {}).get('value', {})
                    fields.append({**base, 'field_name': field, 'value': entered, 'region_id': rid,
                                   'reference_region_ids': [], 'reference_value': ref,
                                   'reference_match': match,
                                   'comparison': 'ambiguous' if status == 'ambiguous' else 'unanswered' if status == 'unanswered' else 'no reference' if ref is None else 'match' if match else 'different',
                                   'status': status, 'region_source': None,
                                   'geometry': {k: box[k] for k in ('x','y','width','height','rotation') if k in box} or None})
            response_rows.extend(fields)
            statuses = {x['status'] for x in fields}
            response_status = 'ambiguous' if 'ambiguous' in statuses else 'unanswered' if statuses == {'unanswered'} else 'partial' if 'unanswered' in statuses else 'answered'
            annotation_rows.append({**base, 'task_id': task.get('id'), 'status': 'cancelled' if cancelled else 'submitted',
                                    'response_status': response_status, 'lead_time': annotation.get('lead_time'),
                                    'created_at': annotation.get('created_at'), 'updated_at': annotation.get('updated_at'),
                                    'completed_at': annotation.get('completed_at') or annotation.get('created_at')})
    item_cols = ['example_id','task_id','text','image','ocr_line_count','task_exported']
    items = pd.DataFrame(task_rows, columns=item_cols)
    if reference is not None:
        items = reference[['example_id']].merge(items, on='example_id', how='left', validate='one_to_one')
        items['task_exported'] = items['task_exported'].fillna(False).astype(bool)
        outside = seen - set(reference['example_id'])
        if outside:
            warnings.warn(f'{len(outside)} exported documents are absent from the reference manifest.', stacklevel=2)
    else:
        warnings.warn('No reference manifest supplied; coverage describes exported documents only.', stacklevel=2)
    ann_cols = ['example_id','annotation_id','annotator_id','task_id','status','response_status','lead_time','created_at','updated_at','completed_at']
    annotations = pd.DataFrame(annotation_rows, columns=ann_cols)
    annotations = add_annotation_sequence(flag_timing(annotations))
    annotations['in_manifest'] = annotations['example_id'].isin(items['example_id'])
    counts = annotations[annotations['status'].eq('submitted') & annotations['in_manifest']].groupby('example_id').size()
    items['annotation_count'] = items['example_id'].map(counts).fillna(0).astype(int)
    items.attrs['manifest_supplied'] = reference is not None
    ext_cols = ['example_id','annotation_id','annotator_id','field_name','value','region_id','reference_region_ids','reference_value','reference_match','comparison','status','region_source','geometry']
    extractions = pd.DataFrame(response_rows, columns=ext_cols)
    extractions['reference_match'] = extractions['reference_match'].astype('boolean')
    return items, annotations, extractions




def show_images(items, example_ids, image_root="../../data", columns=3):
    """Display images from Label Studio local-file references."""

    if isinstance(example_ids, str):
        example_ids = [example_ids]

    examples = items.set_index("example_id").loc[example_ids]

    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(4 * columns, 5 * rows),
        squeeze=False
    )

    for ax, (example_id, item) in zip(axes.flat, examples.iterrows()):
        # Convert the Label Studio URL to a local file path
        image_url = item["image"]
        relative_path = parse_qs(urlparse(image_url).query)["d"][0]
        image_path = Path(image_root) / relative_path

        with Image.open(image_path) as image:
            ax.imshow(image.convert("RGB"))

        ax.set_title(example_id)
        ax.axis("off")

    for ax in axes.flat[len(examples):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

