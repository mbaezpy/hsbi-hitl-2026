# Lab 07 — Annotation quality and disagreement

From the repository root, prepare the complete RTE and Emotion datasets and generate an inspection report:

```bash
python labs/Lab07/prepare_data.py
```

Emotion has one row per worker, item and emotion dimension; seven rows may describe one underlying judgment. RTE includes both premise and hypothesis. The report explains reference coverage and which proposed analyses are feasible.

**Sources:** Snow, O’Connor, Jurafsky and Ng, [“Cheap and Fast — But is it Good?”](https://aclanthology.org/D08-1027/) (EMNLP 2008); The dataset are downloaded from the [extended data release on coauthor Brendan O’Connor’s site](https://brenocon.com/). 