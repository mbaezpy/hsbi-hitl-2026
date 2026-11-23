#!/usr/bin/env python3
"""Download and inspect Snow et al.'s full RTE and emotion annotation release."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import sys
import tarfile
from urllib.error import URLError
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
ARCHIVE = DATA / 'raw' / 'snow2008_extended_201904.zip'
HASH = '4e20418f2e5590332200d34f03fe9dedca2bd44ea1dbca5825499002bdba8821'
URLS = [
    'https://brenocon.com/snow2008_mturk_data_with_orig_files_assembled_201904.zip',
    'https://sites.google.com/site/nlpannotations/snow2008_mturk_data_with_orig_files_assembled_201904.zip',
]
DIMENSIONS = ('anger', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'valence')


def valid_archive(path):
    return path.is_file() and sha256(path.read_bytes()).hexdigest() == HASH


def download():
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    if valid_archive(ARCHIVE):
        print('Using cached verified archive:', ARCHIVE)
        return
    if ARCHIVE.exists():
        raise ValueError(f'Cached archive has an unexpected SHA-256: {ARCHIVE}. Remove it to retry.')
    errors = []
    for url in URLS:
        try:
            print('Downloading:', url)
            with urlopen(Request(url, headers={'User-Agent': 'Lab07-data-preparation/1.0'}), timeout=45) as response:
                contents = response.read()
            if sha256(contents).hexdigest() != HASH:
                raise ValueError('response did not match the expected archive (possibly a sign-in page)')
            with ZipFile(BytesIO(contents)) as archive:
                if archive.testzip() is not None:
                    raise ValueError('ZIP integrity check failed')
            temp = ARCHIVE.with_suffix('.tmp')
            temp.write_bytes(contents)
            temp.replace(ARCHIVE)
            return
        except (OSError, URLError, ValueError, BadZipFile) as error:
            errors.append(f'{url}: {error}')
    raise RuntimeError('Unable to download the original data archive:\n' + '\n'.join(errors))


def unpack():
    """Copy only exact expected members; preserve the full source ZIP as well."""
    with ZipFile(ARCHIVE) as z:
        def get(suffix):
            members = [name for name in z.namelist() if name.endswith('/' + suffix)]
            if len(members) != 1:
                raise ValueError(f'Archive must contain exactly one {suffix}; found {len(members)}')
            return z.read(members[0])

        archive = get('all_collected_data.tgz')
        selected = {
            'README.txt': get('README.txt'),
            'all_collected_data.tgz': archive,
            'rte/rte1.tsv': get('rte/rte1.tsv'),
            'rte/rte.anno.csv': get('rte/rte.anno.csv'),
            'rte/annotated_test.xml': get('rte/orig/annotated_test.xml'),
            'emotion/affect.tsv': get('affect_emotion/affect.tsv'),
            'emotion/affemotion.anno.csv': get('affect_emotion/affemotion.anno.csv'),
        }
    with tarfile.open(fileobj=BytesIO(archive), mode='r:gz') as tar:
        names = {'rte': 'rte.standardized.tsv', **{d: f'{d}.standardized.tsv' for d in DIMENSIONS}}
        for name, member in names.items():
            matches = [m for m in tar.getmembers() if m.isfile() and Path(m.name).name == member]
            if len(matches) != 1:
                raise ValueError(f'Expected one {member} in the source archive; found {len(matches)}')
            selected[f'{"rte" if name == "rte" else "emotion"}/{member}'] = tar.extractfile(matches[0]).read()
    for name, contents in selected.items():
        dest = DATA / 'raw' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.read_bytes() != contents:
            dest.write_bytes(contents)


def read_tsv(path):
    return pd.read_csv(path, sep='\t', dtype=str, keep_default_na=False, encoding='utf-8')


def prepare():
    raw = DATA / 'raw'
    names = {'!amt_annotation_ids': 'assignment_id', '!amt_worker_ids': 'worker_id',
             'orig_id': 'item_id', 'response': 'response', 'gold': 'reference_label'}
    rte = read_tsv(raw / 'rte/rte.standardized.tsv').rename(columns=names)
    source = read_tsv(raw / 'rte/rte1.tsv').rename(columns={'id': 'item_id', 'value': 'source_reference'})
    if source.item_id.duplicated().any():
        raise ValueError('RTE source text has repeated item IDs; cannot safely join.')
    rte = rte.merge(source, how='left', on='item_id', validate='many_to_one', sort=False, indicator=True)
    rte['content_joined'] = rte.pop('_merge').eq('both')
    # The supplementary original CSV covers only some standardized RTE rows;
    # retain its comments and assignment IDs when the worker/item pair exists.
    supplemental = pd.read_csv(raw / 'rte/rte.anno.csv', dtype=str, keep_default_na=False)
    supplemental = supplemental.rename(columns={'id': 'item_id', '!amt_worker_ids': 'worker_id',
                                                '!amt_assignment_ids': 'source_assignment_id', '!comment': 'source_comment'})
    if supplemental.duplicated(['item_id', 'worker_id']).any():
        raise ValueError('Supplementary RTE records repeat worker/item pairs; cannot safely join.')
    rte = rte.merge(supplemental[['item_id', 'worker_id', 'source_assignment_id', 'source_comment']],
                    how='left', on=['item_id', 'worker_id'], validate='many_to_one', sort=False)
    parts = []
    content = read_tsv(raw / 'emotion/affect.tsv').rename(columns={'id': 'item_id', 'headline': 'text'})
    if content.item_id.duplicated().any():
        raise ValueError('Emotion source text has repeated item IDs; cannot safely join.')
    for dimension in DIMENSIONS:
        frame = read_tsv(raw / f'emotion/{dimension}.standardized.tsv').rename(columns=names)
        frame.insert(0, 'dimension', dimension)
        parts.append(frame)
    emotion = pd.concat(parts, ignore_index=True)
    emotion = emotion.merge(content, how='left', on='item_id', validate='many_to_one', sort=False, indicator=True)
    emotion['content_joined'] = emotion.pop('_merge').eq('both')
    # Empty source fields remain empty in the CSV. Numeric conversions validate,
    # never drop or impute rows; an empty rating remains missing.
    for frame in (rte, emotion):
        for column in ('response', 'reference_label'):
            frame[column + '_raw'] = frame[column]
            frame[column] = pd.to_numeric(frame[column].replace('', pd.NA), errors='raise').astype('Int64')
    if len(rte) != len(read_tsv(raw / 'rte/rte.standardized.tsv')) or len(emotion) != sum(len(p) for p in parts):
        raise ValueError('Row count changed during preparation.')
    out = DATA / 'processed'
    out.mkdir(parents=True, exist_ok=True)
    rte.to_csv(out / 'rte.csv', index=False)
    emotion.to_csv(out / 'emotion.csv', index=False)
    return rte, emotion


def spread(counts):
    return f'min {int(counts.min())}, median {counts.median():g}, max {int(counts.max())}' if len(counts) else 'no records'


def inspect(frame, title, group_columns):
    groups = frame.groupby(group_columns, dropna=False).size()
    workers = frame.groupby('worker_id', dropna=False).size()
    valid_reference = frame.reference_label.notna()
    ref_by_worker = frame[valid_reference].groupby('worker_id').item_id.nunique().reindex(frame.worker_id.unique(), fill_value=0)
    gold_conflicts = int((frame.groupby(group_columns, dropna=False).reference_label.nunique(dropna=True) > 1).sum())
    missing = {name: int((frame[name].isna() | frame[name].astype(str).str.strip().eq('')).sum()) for name in frame.columns}
    dups = int(frame.duplicated(subset=['worker_id', *group_columns, 'response']).sum())
    overlap = int((groups >= 2).sum())
    lines = [f'## {title}', '',
        f'- Items: {frame.item_id.nunique()}; annotators: {frame.worker_id.nunique()}; annotation records: {len(frame):,}.',
        f'- Records per {"item and dimension" if len(group_columns) > 1 else "item"}: {spread(groups)}. Worker records: {spread(workers)}.',
        f'- Groups with at least two records: {overlap:,} of {len(groups):,}.',
        f'- Reference coverage: {int(valid_reference.sum()):,}/{len(frame):,} records; conflicting references in {gold_conflicts} groups.',
        f'- Distinct reference-labeled items per worker: {spread(ref_by_worker)}; {int((ref_by_worker >= 20).sum())} of {len(ref_by_worker)} workers have at least 20. This is a descriptive threshold, not proof of adequate qualification.',
        f'- Original content joined for {int(frame.content_joined.sum()):,}/{len(frame):,} records.',
        f'- Missing values by column: {missing}.',
        f'- Duplicate worker/item/response records' + (' per dimension' if len(group_columns) > 1 else '') + f': {dups}. Rows are retained.',
        f'- Repeated worker/item combinations' + (' per dimension' if len(group_columns) > 1 else '') + f": {int(frame.duplicated(subset=['worker_id', *group_columns]).sum())}.",
    ]
    if title == 'RTE':
        lines += [f"- Response distribution: {', '.join(f'{key}: {int(value)}' for key, value in frame.response.value_counts(dropna=False).sort_index().items())}.",
                  f"- Supplementary original RTE annotation CSV covers {int(frame.source_assignment_id.notna().sum()):,}/{len(frame):,} worker/item rows; missing supplementary comments are retained.",
                  '- Reference labels are supplied source labels, not unquestionable facts. Reference-based accuracy is possible where labels are present.']
    else:
        lines.append('- Reference scores are supplied source judgments; they are not objective truth. Exact-match accuracy is usually a poor fit for numerical emotion ratings.')
    return lines


def report(rte, emotion):
    out = ['# Lab 07 dataset inspection', '',
           'The complete standardized annotations from Snow et al. (2008) are retained, including any missing values or repeated records. Item and worker IDs are source IDs. Counts describe annotation records, not independent worker identities across studies.', '',
           'Sources: [Snow et al. (2008)](https://aclanthology.org/D08-1027/); [author-hosted extended release](https://brenocon.com/); [TrentoCrowdAI catalogue](https://github.com/TrentoCrowdAI/crowdsourced-datasets). The former catalogue download URL redirects to sign-in. The original release does not state a general license; check the original dataset and underlying text terms before reuse or redistribution.', '']
    out += inspect(rte, 'RTE', ['item_id']) + ['']
    out += inspect(emotion, 'Emotion', ['item_id', 'dimension']) + ['',
        f"Distinct worker–headline pairs: {len(emotion[['worker_id', 'item_id']].drop_duplicates()):,}; these yield {len(emotion):,} dimension-level records.",
        '',
        'The supplementary original Emotion CSV contains blank rating columns; numerical ratings and source reference scores come from the seven standardized files. Both source files are kept under `data/raw/`.', '',
        'Emotion consists of ratings of the same headlines on seven dimensions; each row is one item/worker/dimension response, not necessarily a separate annotation session. Six emotion scales are 0–100; valence is −100–100 ([original paper](https://aclanthology.org/D08-1027/)). Observed ranges and coverage:', '',
        '| Dimension | Items | Workers | Ratings | Ratings per item (min / median / max) | Observed range | Reference coverage |',
        '|---|---:|---:|---:|---|---|---|']
    for dimension, part in emotion.groupby('dimension', sort=True):
        counts = part.groupby('item_id').size()
        values = part.response.dropna()
        out.append(f'| {dimension} | {part.item_id.nunique()} | {part.worker_id.nunique()} | {len(part):,} | {int(counts.min())} / {counts.median():g} / {int(counts.max())} | {values.min()}–{values.max()} | {part.reference_label.notna().sum()}/{len(part)} |')
    out += ['', 'The seven files cover the same headlines when their item IDs overlap; do not add per-dimension item counts to claim distinct documents. Rating distributions can be compared by item and dimension and used to explore human variation; reference scores are one point of comparison.', '',
        '## Exercise feasibility', '',
        '| Exercise | RTE | Emotion |', '|---|---|---|',
        '| Pairwise agreement | Feasible with shared items; check worker overlap. | Feasible within each dimension; exact agreement is strict for numerical ratings. |',
        '| Krippendorff’s alpha | Feasible: nominal binary labels. | Feasible per dimension with interval distance; justify the measurement level. |',
        '| Item disagreement and worker comparisons | Feasible with individual responses; worker workloads vary. | Feasible per dimension; compare distributions and worker overlap. |',
        '| Majority voting and soft labels | Feasible for binary categories. | Exact-score distributions are feasible; majority voting on fine-grained numbers may be uninformative. |',
        '| Reliability weighting and Dawid–Skene | Feasible for categorical RTE, subject to modeling assumptions and worker overlap. | Standard categorical Dawid–Skene is a poor fit for numerical scores; use an appropriate continuous model or justify discretization. |',
        '| Known-answer qualification and before/after filtering | Partially feasible using reference-labeled items and held-out judgments; avoid using evaluation items for qualification. Varying worker coverage limits qualification-set size. | Partially feasible using reference scores as imperfect comparison targets and a justified numerical error measure; subjectivity makes a pass/fail gold standard questionable. |', '',
        'Do not infer qualification viability from total row counts alone: choose held-out items, inspect per-worker overlap and reference coverage, and ensure enough independent evaluation observations remain. These checks do not perform an experiment.', '']
    (DATA / 'dataset_report.md').write_text('\n'.join(out), encoding='utf-8')


def main():
    download()
    unpack()
    rte, emotion = prepare()
    report(rte, emotion)
    print(f'Prepared {len(rte):,} RTE and {len(emotion):,} emotion-dimension records.')
    print('Report:', DATA / 'dataset_report.md')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError, BadZipFile, tarfile.TarError, pd.errors.ParserError) as error:
        sys.exit(f'Data preparation failed: {error}')
