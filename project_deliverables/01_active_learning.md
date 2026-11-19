# Project Deliverable 1 — Active Learning

**Deadline:** Friday, 27 November 2026 (hard deadline)  
**Submission:** ILIAS -> Project Deliverables

## Goal

This first deliverable brings together the Active Learning experiments developed during Labs 03–05.

The goal is to **present and evaluate your experimental results**, comparing different acquisition strategies and discussing whether Active Learning can improve labeling efficiency in your project.

You should consolidate your work into a clear, reproducible experiment with results and conclusions that can be reviewed without navigating through multiple notebooks.

## 1. What your results should cover

Following the requirements introduced in the labs, your final comparison should include four acquisition strategies:

- **Random sampling:** The baseline for comparison.
- **Uncertainty sampling:** At least one uncertainty-based acquisition function appropriate for your task.
- **Diversity sampling:** At least one diversity-based acquisition strategy.
- **Representative sampling:** At least one representativeness-based acquisition strategy.

Your results should be based on:

- Multiple Active Learning rounds, using reference labels as a simulated annotation oracle.
- Equivalent experimental conditions across strategies, including the initial labeled set, annotation budget, model, and evaluation data.
- **At least five repetitions**, using different random seeds or initial labeled sets.
- Learning curves showing **mean performance and variability** across repetitions, using an evaluation metric appropriate for your task.

You do not need to implement every method covered in the labs. Focus on the approaches you selected for your project and briefly justify your choices.

Additional experiments and configurations are welcome but not required.

## 2. Results and discussion

Your main results should be presented in a Jupyter notebook, including:

- A brief description of the experimental setup: dataset, model, acquisition strategies, initial labeled set, annotation budget, and evaluation metric.
- Learning curves comparing the required strategies, with variability across repetitions.
- A short discussion of the results: Which strategies perform better? Do they outperform random sampling? What does this tell you about the potential value of Active Learning for your project?
- Relevant limitations of your experiment and conclusions.

**Negative results are equally valid.** Active Learning does not necessarily outperform random sampling. What matters is that you conduct a meaningful comparison and interpret the findings.

You do not need to prepare a separate report. Include your explanations and conclusions directly in the notebook using Markdown cells.

## 3. GitHub repository

Create a **GitHub repository for your semester project**. You will continue using and extending this repository for future project deliverables.

You are free to organize the repository according to your project's needs. However, we expect:

- **A clear entry point for each deliverable.** For this first deliverable, identify one main Jupyter notebook containing your Active Learning experiments, results, and conclusions. You may use additional notebooks and scripts, but we should not need to search through multiple files to find your final results.
- **A README.md at the repository root.** Include a short project description, team members, and instructions for installing dependencies, obtaining required datasets or models, and running the project. The README should also link to the main notebook for each deliverable.
- **Visible results.** Commit your main notebook with its outputs, especially the learning curves, so we can review the results without rerunning every experiment.

You do not need to upload large datasets, trained models, or generated files to GitHub. Provide instructions for obtaining them when necessary.

**Important — Existing repositories**

If you have already created a GitHub repository organized around your Active Learning experiments, **you do not need to reorganize it for this submission**.

The structure below is only a suggestion for organizing your semester project over time. You can reorganize or extend your repository for future deliverables.

For now, what matters is that **your code and results are clearly organized, the main experiment is easy to find, and the instructions for running it are available**. There is no need to spend time restructuring a repository that already works.


### Suggested organization (optional)

For example, you could organize your repository as follows:

```text
hitl2026-<project-name>/
├── README.md
├── requirements.txt
├── active_learning/
│   └── experiments.ipynb
├── annotation/
├── data/
└── src/
```

This is only an example. You may use a different structure as long as the deliverables are easy to locate and the project can be reproduced.

## 4. Submission through ILIAS

Submit your work in the **Project Deliverables** section in ILIAS.

Only one submission per team is required. Include:

- Project name and short description.
- Team members.
- GitHub repository URL.
- Direct link to your main Active Learning notebook.
- Git commit hash identifying the submitted version.

Make sure the repository is accessible and all required work is committed before the deadline.

**No additional PDF, ZIP file, or separate report is required.**

## References

The following course labs provide the foundation for this deliverable:

- [Lab 03 — Uncertainty Sampling](../labs/Lab03/active_learning_uncertainty.ipynb)
- [Lab 04 — Diversity and Representative Sampling](../labs/Lab04/al_diversity_rep.ipynb)
- [Lab 05 — Acquisition Strategies and Policies](../labs/Lab05/al_policy_framework.ipynb)

You may reuse and adapt the methods, code, and experimental framework introduced in these notebooks to your own project.