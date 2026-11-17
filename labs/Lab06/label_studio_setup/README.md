# Label Studio — Quickstart Guide

**Lab 06 · Designing and Piloting an Annotation Task**

In this lab, you will turn your Annotation Design Canvas into a working task in Label Studio, invite someone to test it, and export the results for pilot analysis.

This guide uses Label Studio Community Edition. Some collaboration and quality-management features are only available in the Enterprise version.

## 1. Accounts and access

Label Studio accounts belong to the specific installation you are using. They are not global accounts shared across Label Studio servers.

One account can access multiple projects on the same instance.

**Important limitation:** In Community Edition, all registered users can access all projects. Do not upload sensitive or private data that other users should not see.

## 2. Create a project

1. Open Label Studio and select **Create Project**.
2. Enter a descriptive project name.
3. Import your pilot dataset, or import it later from the project.
4. Select a labeling template matching your task, or configure one manually.
5. Save your project.

For our banking example, use **text classification**.

Your CSV should contain the information that the annotator needs, for example:

| example_id | text |
|---|---|
| bank_001 | I forgot my passcode. |
| bank_002 | My transfer is still pending. |

Keep existing reference or ground-truth labels in a separate file. Do not show them to the pilot annotators.

## 3. Configure the labeling interface

Go to **Project → Settings → Labeling Interface**.

You can start from a template or select **Code** to customize the interface.

Label Studio uses an XML-like configuration language. The most important elements for our banking example are:

| Element | Purpose |
|---|---|
| `Text` with `value="$text"` | Displays the imported text field |
| `Choices` | Defines the response control |
| `name="label"` | Identifies this annotation field in the results |
| `toName="text"` | Connects the response control to the displayed text |
| `choice="single"` | Allows one selected option |
| `Choice value` | Text displayed to annotators |
| `Choice alias` | Internal value saved in annotation results |
| `Choice hint` | Description displayed on hover |

For example, a readable option such as **Bank transfer — Pending** can use the alias `bank_transfer_pending` to match the labels expected by your ML pipeline.

You can also use the `html` attribute of a `Choice` to display formatted descriptions directly under its label.

````xml
  <Choice value="Account access - Forgotten passcode"
        alias="account_access_passcode"
        html="&lt;div&gt;&lt;strong&gt;Forgotten passcode&lt;/strong&gt;&lt;div style='font-size:12px;color:#666'&gt;Customer has forgotten their passcode or cannot access their account because of it.&lt;/div&gt;&lt;/div&gt;"/>
    ````

````xml
    <Choice value="Account access - Forgotten passcode"
            alias="account_access_passcode"
            hint="The customer has forgotten their passcode or cannot access their account because of it."/>
````


**Design tip:** Make the item being annotated easy to find. Avoid letting instructions, definitions, or decorative interface elements dominate the workspace.

## 4. Provide annotation instructions

Go to **Project → Settings → Annotation → Labeling Instructions**.

The instruction field supports basic HTML formatting.

Your instructions should explain:

- The purpose of the task and what receives one annotation.
- How to make the judgment.
- The meaning of the labels or fields.
- Important decision rules.
- Typical and difficult examples.
- What to do when information is missing or no answer fits.

**Show before labeling** displays the instructions in a pop-up when annotators start.

For longer instructions, this pop-up can become difficult to read. An alternative is to keep a brief introduction visible in the labeling interface and provide detailed definitions and examples in expandable panels.

For our banking example, we use separate sections for **Guidelines and decision rules**, **Label definitions**, and **Examples and difficult cases**.

## 5. Invite annotators

To invite someone:

1. Open the main menu and select **Organization**.
2. Select **Add Members**.
3. Copy the invitation link and share it.
4. Ask the person to register and open your project.

**Community Edition limitation:** Invitations are not project-specific. Invited users can access other projects on the same instance.

For this lab, invite a classmate to complete a small pilot. Ask them to use the instructions independently before explaining the task to them.

## 6. Organize annotation work

Community Edition does not provide built-in worker-specific assignment and quota management comparable to dedicated crowdsourcing platforms.

A practical workaround is to use **Data Manager tabs**.

### Create an annotation batch

1. Open your project's Data Manager.
2. Create a new tab using the **+** control.
3. Rename the tab for a particular annotator or batch.
4. Add a filter selecting the intended items.
5. Ask the annotator to open that tab and choose **Label Tasks As Displayed**.

For example, you could use filters based on the task ID or an imported batch identifier.

**Important:** The task ID generated by Label Studio is not necessarily the same as the `example_id` in your original dataset. Verify the actual IDs before defining ranges.

These tabs organize the work, but do not restrict access. Anyone can still open another tab or the full project.

### Multiple annotations per task

A Label Studio task can contain several independent annotations. By default, a task is considered completed after one annotation.

To collect multiple annotations per task, we need to change the `maximum_annotations` setting. Unfortunately, this setting is not directly available in the Label Studio Community Edition interface.

To make this easier, we provide an interactive script at the repository root:

`scripts/configure_lstudio.py`

Run it from the repository root:

```bash

export LSTUDIO_URL=<url of labelstudio>
export LSTUDIO_PT=<personal token>

python scripts/configure_lstudio.py
```

The script uses the Label Studio API to list available projects, select a project, and change the **Annotations per task** setting.

Set this value to the number of independent annotations you want for each example (e.g., `2` if each example should be annotated by two people).

**Note:** The script requires a Label Studio Personal Access Token in the `LSTUDIO_PT` environment variable.

## 7. Monitor progress

The Data Manager allows you to filter and inspect annotation progress.

Useful views include:

- **Not yet annotated:** Items with no submitted annotations.
- **Annotated:** Items with at least one submitted annotation.
- **Completed:** Items that have reached the project's required annotation count.

For example, if two annotations are required, a task with only one annotation is *partially annotated*, not completed.

You can create saved tabs to monitor these groups. Available filter names may depend on the installed Label Studio version.

## 8. Export the pilot results

When the pilot is finished:

1. Open the project Data Manager.
2. Select **Export**.
3. Choose **JSON** (the full/raw format).
4. Download the file.

The raw export contains the task information, individual annotations, annotator identifiers, selected labels, and available metadata such as annotation time.

Use the **Lab 06 Pilot Analysis notebook** to inspect completion, annotation times, reference-label discrepancies, and potentially difficult cases.

Do not aggregate the annotations yet. We will study annotation agreement and aggregation in Lab 07.

## 9. Before finishing the lab

Check that:

- Your annotation task matches the decisions in your Design Canvas.
- Your labels or fields are correctly defined.
- Another person can understand and complete the task.
- You have collected a small pilot sample.
- You have exported the results in raw JSON format.
- You have identified at least one problem or possible improvement.

**Remember:** The objective is not to create a perfect task immediately. It is to design something usable, test it with people, learn from the pilot, and revise it.