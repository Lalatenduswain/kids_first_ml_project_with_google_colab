# 📖 kids_first_ml_project.ipynb — Complete Deep-Dive Explanation

This document explains **every cell, every line of code, and every concept** in the notebook — written for anyone (kids, parents, teachers) who wants to understand exactly what is happening and *why*.

---

## Table of Contents

1. [The Big Picture — What Is This Notebook Doing?](#1-the-big-picture)
2. [Cell 0 — Introduction & Problem Statement](#2-cell-0--introduction--problem-statement)
3. [Step 1 — Importing Libraries](#3-step-1--importing-libraries)
4. [Step 2 — Building the Dataset](#4-step-2--building-the-dataset)
5. [Step 3 — Visualising the Data (EDA)](#5-step-3--visualising-the-data-eda)
6. [Step 4 — Training the AI Model](#6-step-4--training-the-ai-model)
7. [Step 5 — Predicting New Fruits](#7-step-5--predicting-new-fruits)
8. [Step 6 — Victory Charts](#8-step-6--victory-charts)
9. [Glossary Cell — Key ML Terms](#9-glossary-cell--key-ml-terms)
10. [End-to-End Data Flow](#10-end-to-end-data-flow)
11. [Why 100% Accuracy? Is That Normal?](#11-why-100-accuracy-is-that-normal)
12. [What Could You Change? (Experiments)](#12-what-could-you-change-experiments)

---

## 1. The Big Picture

The notebook solves a **binary classification problem**: given a fruit's weight and colour, decide whether it is a **Mango (ଆମ୍ବ)** or an **Orange (କମଳା)**.

```
Input features  →  ML Model  →  Prediction
[weight, color]    Decision     "Mango" or
                   Tree         "Orange"
```

The entire workflow follows the standard Machine Learning pipeline:

```
Collect Data → Visualise → Train Model → Evaluate → Predict New Data
```

Everything runs inside a **single Jupyter Notebook** — no external files, no internet needed, no GPU required. It runs in seconds on any computer, including Google Colab (free).

---

## 2. Cell 0 — Introduction & Problem Statement

```markdown
# 🕵️🥭🍊 ମୋର ପ୍ରଥମ AI Project — ଫଳ ଗୋଇନ୍ଦା!
```

### What it does
This is a **Markdown cell** — it contains no runnable code. It sets the story:

> A fruit shop owner has lost his glasses. He can't tell Mango from Orange. Our AI Robot will solve this!

### Why this matters
A good ML project always starts with a **clear problem statement**. Before writing a single line of code, you must know:
- What question am I answering? → *"Which fruit is this?"*
- What data do I have? → *Weight + Colour*
- What should the output be? → *Mango or Orange (0 or 1)*

### The Robot character
The notebook uses a sassy Robot (🤖) as a narrator. This is purely for fun and readability — it has no effect on the code.

---

## 3. Step 1 — Importing Libraries

### The Markdown header (cell-1)
```markdown
## Step 1 — 🛒 ଦୋକାନରୁ ଜିନିଷ ଆଣ (Get the Tools!)
```
Analogy: *"Just like you take out your toys before playing, we bring out our tools before coding."*

### The Code (cell-2) — Line by Line

```python
import numpy as np
```
- **What is numpy?** A Python library for fast mathematical operations on lists of numbers (called *arrays*).
- **Why do we need it?** Our fruit data is stored as a 2D list. `numpy` converts it into an *array* that scikit-learn can process. It also makes number operations (like averages, comparisons) much faster.
- **`as np`** — a shortcut alias. Instead of writing `numpy.array(...)` every time, we write `np.array(...)`. Saves typing!

```python
import matplotlib.pyplot as plt
```
- **What is matplotlib?** The most popular Python library for drawing charts and graphs.
- **`pyplot`** — a sub-module of matplotlib that works like a painter's canvas. You call functions like `plt.scatter()`, `plt.bar()`, `plt.show()` to draw things.
- **`as plt`** — again a shortcut alias.

```python
from sklearn.tree import DecisionTreeClassifier
```
- **What is sklearn (scikit-learn)?** The most widely-used Python library for Machine Learning. It contains ready-made algorithms you can use without implementing them from scratch.
- **`from sklearn.tree`** — we are going into the `tree` section of sklearn (the section that has tree-based algorithms).
- **`import DecisionTreeClassifier`** — we grab the specific tool called `DecisionTreeClassifier`. This is our AI brain — the thing that will actually *learn* from data.
- **Why DecisionTree?** It works like a flowchart of yes/no questions: *"Is weight > 120g? Yes → Orange. No → Mango."* Very easy to understand and visualise.

```python
from sklearn.metrics import accuracy_score
```
- **What is it?** A function that compares the model's predictions against the correct answers and returns a percentage of how many it got right.
- **Formula:** `accuracy = (correct predictions / total predictions) × 100`

```python
print("=" * 40)
print("🤖 Robot: 'ସବୁ tools ଆସିଗଲା!'")
...
```
- Pure print statements for fun output. The `"=" * 40` prints 40 `=` signs in a row — a simple way to draw a separator line.
- These `print` lines have **zero effect on the ML logic** — they just make the output friendly and readable.

### What the output looks like
```
========================================
🤖 Robot: 'ସବୁ tools ଆସିଗଲା!'
🎒 Backpack ଭର୍ତ୍ତି! Mission start! 🚀
========================================

✅ numpy    — calculator ଭଳି (ଗଣିତ ପାଇଁ)
✅ matplotlib — ଚିତ୍ରକର ଭଳି (chart ଆଁକିବ)
✅ sklearn  — AI ର super brain! 🧠

🎉 READY TO ROLL!
```

---

## 4. Step 2 — Building the Dataset

### The Markdown header (cell-3)
Introduces the two key **features** (clues the AI uses):

| Clue | Mango 🥭 | Orange 🍊 |
|------|----------|-----------|
| Weight | 80–110g (light) | 130–200g (heavy) |
| Colour code | 0 (yellow) | 1 (orange) |

These two facts are all the AI needs. Simple, but powerful.

### The Code (cell-4) — Line by Line

```python
fruits_data = [
    [150, 1],  # Orange
    [170, 1],  # Orange
    ...
    [90,  0],  # Mango
    [100, 0],  # Mango
    ...
]
```
- This is a **Python list of lists**. Each inner list `[weight, colour]` represents one fruit.
- We have **14 fruits total**: 7 oranges + 7 mangoes.
- This is our **dataset** — the "textbook" from which the AI will learn.

**Why these specific numbers?**
- Oranges: 130–200g. In real life, a standard orange weighs ~150–200g.
- Mangoes: 80–110g. A small Indian mango (like Alphonso) can be 80–120g.
- The gap between the two groups (110g mango max vs 130g orange min) means they are **perfectly separable** — important for achieving 100% accuracy.

```python
labels = [1, 1, 1, 1, 1, 1, 1,
          0, 0, 0, 0, 0, 0, 0]
```
- This list maps each fruit in `fruits_data` to its **correct answer** (label).
- Position 0 in `fruits_data` (`[150, 1]`) maps to position 0 in `labels` (`1`) → it's an orange.
- Position 7 in `fruits_data` (`[90, 0]`) maps to position 7 in `labels` (`0`) → it's a mango.
- The **order must match perfectly** between `fruits_data` and `labels`.

```python
X = np.array(fruits_data)
y = np.array(labels)
```
- We convert both Python lists into **numpy arrays**.
- **Convention:** In ML, `X` always means the **features** (inputs), `y` always means the **labels** (answers). This is a universal convention used worldwide.
- `X` is a **2D array** with shape `(14, 2)` — 14 rows (fruits), 2 columns (weight, colour).
- `y` is a **1D array** with shape `(14,)` — 14 answers.

```python
print(f"🥭 Team Mango  (ଆମ୍ବ team) : {sum(y==0)} ଟି")
print(f"🍊 Team Orange (କମଳା team): {sum(y==1)} ଟି")
```
- `y==0` creates a boolean array (`True/False`) — `True` wherever y is 0.
- `sum(y==0)` counts how many `True` values exist → counts mangoes.
- The `f"..."` syntax is an **f-string** — it lets you embed Python expressions inside `{}` directly in a string.

### What the output looks like
```
🗂️  EVIDENCE COLLECTED!
======================================
🥭 Team Mango  (ଆମ୍ବ team) : 7 ଟି
🍊 Team Orange (କମଳା team): 7 ଟି
📦 ମୋଟ suspects           : 14 ଟି
======================================
🕵️  Detective Robot: 'ଚାଲ! Case solve କରିବା!'
```

---

## 5. Step 3 — Visualising the Data (EDA)

**EDA = Exploratory Data Analysis** — looking at your data visually *before* training to understand patterns.

### Why EDA matters
If you can already see a clear separation in a chart, it means the AI should be able to learn it easily. If the data is all jumbled together, expect low accuracy.

### The Code (cell-6) — Line by Line

```python
plt.figure(figsize=(9, 5))
```
- Creates a new empty chart canvas.
- `figsize=(9, 5)` sets the size: 9 inches wide, 5 inches tall.

```python
mango_weights  = [X[i][0] for i in range(len(y)) if y[i] == 0]
orange_weights = [X[i][0] for i in range(len(y)) if y[i] == 1]
```
- **List comprehensions** — a compact Python way to build a list using a loop.
- `X[i][0]` = the weight of the i-th fruit (column 0).
- `if y[i] == 0` = only include it if it's a mango.
- Result: two separate lists of weights — one for mangoes, one for oranges.

```python
plt.scatter(range(len(mango_weights)), mango_weights,
            color='gold', s=300, label='🥭 ଆମ୍ବ (Slim guys!)', zorder=3,
            edgecolors='darkorange', linewidths=2)
```
- `plt.scatter()` draws a **scatter plot** — dots on a graph.
- `range(len(mango_weights))` = x-axis positions: 0, 1, 2, 3, 4, 5, 6 (the 7 mango positions).
- `mango_weights` = y-axis positions (the actual gram values).
- `color='gold'` = yellow dots for mangoes.
- `s=300` = size of each dot (300 units² — big enough to see clearly).
- `label=...` = text shown in the legend.
- `zorder=3` = drawing order. Higher number = drawn on top of lower numbers. Ensures dots appear above the grid lines.
- `edgecolors='darkorange'` + `linewidths=2` = adds a dark orange border around each dot, making it look polished.

```python
plt.axhline(y=120, color='red', linestyle='--', alpha=0.5, linewidth=2,
            label='🚔 Boundary (120g)')
```
- `axhline` = draws a **horizontal line** across the entire chart.
- `y=120` = the line is at 120 grams — the boundary between the two fruit groups.
- `linestyle='--'` = dashed line.
- `alpha=0.5` = 50% transparent (so it doesn't overpower the dots).
- This line visually shows *where the Decision Tree will draw its boundary*.

```python
plt.title(...)
plt.ylabel(...)
plt.xlabel(...)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```
- Standard matplotlib commands to label and display the chart.
- `plt.grid(True, alpha=0.3)` — adds faint background grid lines.
- `plt.tight_layout()` — auto-adjusts spacing so nothing gets clipped.
- `plt.show()` — renders and displays the chart.

### What you see
A scatter plot with:
- 🟡 Gold dots clustered **below 120g** — the 7 mangoes
- 🟠 Dark orange dots clustered **above 120g** — the 7 oranges
- A red dashed line at 120g separating both groups perfectly

This visual proves the data is **linearly separable** — a straight line can perfectly divide the two classes.

---

## 6. Step 4 — Training the AI Model

### The Markdown header (cell-7)
Explains the analogy:
- You learn by eating food (experience over time).
- AI learns by consuming data (training in milliseconds).

### The Code (cell-8) — Line by Line

```python
import time
```
- Imported but not actually used for timing in this cell (it's imported for potential future use or just for show with the loading animation joke).

```python
model = DecisionTreeClassifier(random_state=42)
```
- Creates a **Decision Tree** model object. At this point it hasn't learned anything yet — it's like a blank-brain robot.
- `random_state=42` — sets a **random seed**. Decision Trees can have some randomness when choosing splits. Fixing the seed ensures you get the **exact same result every time** you run the code. (42 is a popular choice in ML — it's a pop culture reference to *The Hitchhiker's Guide to the Galaxy*.)

**What is a Decision Tree?**
It learns a flowchart of yes/no questions from data:
```
Is weight > 120g?
├── YES → Orange 🍊
└── NO  → Mango 🥭
```
For this dataset, one single question is enough because the two groups don't overlap at all.

```python
model.fit(X, y)
```
- **This is the most important line in the entire notebook.**
- `.fit()` = **training**. The model looks at all 14 fruits (`X`) and their correct answers (`y`), and learns the pattern.
- Internally, it tries every possible split point (e.g., "weight > 80?", "weight > 85?", ... "weight > 200?") and picks the one that best separates mangoes from oranges.
- It finds: *"weight > 120 perfectly separates everything"* → that becomes the tree.
- This takes less than 1 millisecond on a modern computer.

```python
predictions = model.predict(X)
```
- `.predict()` asks the trained model to classify the same 14 fruits it was trained on.
- Returns an array like `[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]`.
- Note: we're testing on **training data** here (the same data we learned from). This is why accuracy is 100% — see section 11 for a deeper explanation.

```python
accuracy = accuracy_score(y, predictions) * 100
```
- `accuracy_score(y, predictions)` compares:
  - `y` = the correct answers: `[1,1,1,1,1,1,1,0,0,0,0,0,0,0]`
  - `predictions` = the model's answers: `[1,1,1,1,1,1,1,0,0,0,0,0,0,0]`
- It counts matching positions and divides by total: `14/14 = 1.0`
- Multiply by 100 → `100.0` (percent)

```python
print(f"📝 AI ର Marks   : {accuracy:.0f}% 📊")
```
- `{accuracy:.0f}` formats the number with 0 decimal places. So `100.0` becomes `100`.

```python
if accuracy == 100:
    print("🎉 ୧୦୦%!! AI ଟି TOPPER!!")
    print("🤖 Robot: 'ଶିକ୍ଷକ ମୋତେ Gold Medal ଦିଅ! 🥇'")
    print("🏫 School: 'ତୁ Robot... Medal ନ ମିଳିବ 😤'")
    print("🤖 Robot: '...okay fine 😞'")
```
- A fun conditional block: if perfect score, the Robot brags and gets roasted. Otherwise, it asks for more data.

### What the output looks like
```
🔌 AI brain ଚାଲୁ ହେଉଛି...
⚡ Loading... [██████████] 100% — DONE! (actually instant 😅)

🎓 TRAINING REPORT CARD
===================================
📝 AI ର Marks   : 100% 📊
📚 Questions    : 14 ଟି ଫଳ
✅ Correct      : 14 ଟି
===================================

🎉 ୧୦୦%!! AI ଟି TOPPER!!
🤖 Robot: 'ଶିକ୍ଷକ ମୋତେ Gold Medal ଦିଅ! 🥇'
🏫 School: 'ତୁ Robot... Medal ନ ମିଳିବ 😤'
🤖 Robot: '...okay fine 😞'
```

---

## 7. Step 5 — Predicting New Fruits

### The Markdown header (cell-9)
Introduces 4 **mystery fruits** — fruits the model has *never seen before*. This simulates real-world usage: the model was trained, now a customer walks in with an unknown fruit.

### The Code (cell-10) — Line by Line

```python
new_fruits = [
    [155, 1],   # Mystery Fruit A — heavy + orange colour
    [ 92, 0],   # Mystery Fruit B — light  + yellow colour
    [175, 1],   # Mystery Fruit C — very heavy + orange colour
    [ 88, 0],   # Mystery Fruit D — light  + yellow colour
]
```
- 4 new fruits with `[weight, colour]` format — same as training data.
- These are **not in the training set** — the model has never seen them.
- The expected answers are obvious to us (A and C are oranges, B and D are mangoes), but the model doesn't know — it has to figure it out from what it learned.

```python
fruit_names  = ['A', 'B', 'C', 'D']
fruit_emojis = ['🎭', '🥸', '🤔', '😶']
label_map    = {0: '🥭 ଆମ୍ବ (MANGO!)', 1: '🍊 କମଳା (ORANGE!)'}
```
- `fruit_names` — letters used for display.
- `fruit_emojis` — funny emojis for each mystery fruit (masked/disguised theme).
- `label_map` — a Python **dictionary** that converts the numeric prediction (0 or 1) into a human-readable string.

```python
results = model.predict(np.array(new_fruits))
```
- `np.array(new_fruits)` converts the list to a numpy array (required by sklearn).
- `model.predict(...)` runs each of the 4 fruits through the learned Decision Tree.
- The tree asks: *"Is weight > 120?"*
  - Fruit A: 155g → YES → Orange ✅
  - Fruit B: 92g  → NO  → Mango ✅
  - Fruit C: 175g → YES → Orange ✅
  - Fruit D: 88g  → NO  → Mango ✅

```python
for i, (fruit, result) in enumerate(zip(new_fruits, results)):
    color_name = 'ହଳଦିଆ' if fruit[1] == 0 else 'କମଳା'
    print(f"{fruit_emojis[i]} Fruit {fruit_names[i]}:  {fruit[0]}g  {color_name:>6}  →  {label_map[result]}")
```
- `zip(new_fruits, results)` — pairs each fruit with its prediction result.
- `enumerate(...)` — also gives an index `i` (0, 1, 2, 3) for the emoji and name.
- `fruit[1]` — the colour code (second element of each fruit list).
- `{color_name:>6}` — right-aligns `color_name` in a field of 6 characters (for neat column alignment).
- `label_map[result]` — converts 0/1 into the Odia+English name.

### What the output looks like
```
🔍 DETECTIVE REPORT — MYSTERY FRUITS EXPOSED! 🎉
================================================
Fruit    ଓଜନ   ରଙ୍ଗ  →  ପରିଚୟ
------------------------------------------------
🎭 Fruit A:  155g  କମଳା  →  🍊 କମଳା (ORANGE!)
🥸 Fruit B:   92g  ହଳଦିଆ  →  🥭 ଆମ୍ବ (MANGO!)
🤔 Fruit C:  175g  କମଳା  →  🍊 କମଳା (ORANGE!)
😶 Fruit D:   88g  ହଳଦିଆ  →  🥭 ଆମ୍ବ (MANGO!)
================================================

🤖 Robot: 'ମୁଁ ସବୁ ଚିହ୍ନଟ କଲି! ଏବେ ଆଇସ୍ ଖିଆ ମିଳିବ? 🍦'
👨‍🔬 Scientist: 'ତୁ ଯନ୍ତ୍ର... ଆଇସ୍ ଖୁଆ ମୁ! 😂'
```

---

## 8. Step 6 — Victory Charts

### The Code (cell-12) — Line by Line

```python
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
```
- Creates a **figure with 2 subplots side by side** (1 row, 2 columns).
- `fig` = the entire figure object (the big canvas).
- `axes` = an array of 2 subplot objects. `axes[0]` = left chart, `axes[1]` = right chart.
- `figsize=(13, 5)` = 13 inches wide, 5 inches tall.

```python
fig.suptitle('🏆 AI Detective — Mission Accomplished! 🕵️', fontsize=16, fontweight='bold', y=1.02)
```
- `suptitle` = **super title** — a title for the entire figure (above both subplots).
- `y=1.02` = places it slightly above the normal position so it doesn't overlap.

#### Chart 1 — Pie Chart

```python
explode = (0.05, 0.05)
axes[0].pie(
    [7, 7],
    labels=['🥭 Team Mango\n(ଆମ୍ବ — 7 ଟି)', '🍊 Team Orange\n(କମଳା — 7 ଟି)'],
    colors=['gold', 'darkorange'],
    autopct='%1.0f%%',
    startangle=90,
    explode=explode,
    shadow=True,
    textprops={'fontsize': 12}
)
```
- `[7, 7]` = the two slices are equal (7 mangoes, 7 oranges → 50/50 split).
- `autopct='%1.0f%%'` = automatically adds percentage labels on each slice (`%1.0f` = integer format, `%%` = literal `%` sign).
- `startangle=90` = rotates the pie 90° so the first slice starts at the top (12 o'clock position).
- `explode=(0.05, 0.05)` = each slice is pulled out 5% from the centre — a "dramatic" visual effect.
- `shadow=True` = adds a drop shadow for a 3D-ish look.

#### Chart 2 — Accuracy Bar Chart

```python
bar = axes[1].bar(
    ['🤖 AI ର Score'], [accuracy],
    color='mediumseagreen', width=0.45,
    edgecolor='darkgreen', linewidth=2
)
```
- `axes[1].bar(x_labels, heights)` — draws a **bar chart**.
- Single bar with label `'🤖 AI ର Score'` and height = `accuracy` (which is 100).
- `width=0.45` = bar is slightly less than full width (looks cleaner).

```python
axes[1].set_ylim(0, 115)
```
- Sets y-axis range from 0 to 115. Why 115 and not 100? To leave space above the bar for the `100% 🥳` text label — otherwise it would be clipped.

```python
axes[1].text(0, accuracy + 3, f'{accuracy:.0f}% 🥳',
             ha='center', fontsize=18, fontweight='bold', color='darkgreen')
```
- Adds a text annotation directly on the chart.
- Position: x=0 (centre of bar), y=accuracy+3 (3 units above the top of the bar).
- `ha='center'` = horizontally centred.

```python
axes[1].axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect score')
```
- Draws a horizontal dashed line at y=100 — a reference line showing where "perfect" is.

---

## 9. Glossary Cell — Key ML Terms

The final markdown cell (cell-13) contains a **three-column table** explaining ML terms:

| ML Term | Real Meaning | Funny Analogy |
|---|---|---|
| **Dataset** | The 14 fruits with features | AI ର Tiffin box 🍱 |
| **Training** | Showing data to the model | Robot School 🏫 |
| **Model** | The learned Decision Tree | Robot ର brain 🤖 |
| **Prediction** | The model's output | Robot ର guess 🎯 |
| **Accuracy** | % of correct predictions | Report Card 📝 |
| **Overfitting** | Memorising without understanding | Robot rote-learns 😂 |

**Overfitting** is hinted at here but not demonstrated — it's an important concept for future learning. See section 11 below.

---

## 10. End-to-End Data Flow

Here is how data flows through the entire notebook:

```
Step 2: Raw Python list
fruits_data = [[150,1],[170,1],...,[90,0],[80,0],...]
labels      = [1,1,1,1,1,1,1,0,0,0,0,0,0,0]
        ↓
np.array() conversion
X.shape = (14, 2)    ← 14 fruits, 2 features each
y.shape = (14,)      ← 14 correct answers
        ↓
Step 3: EDA (just reading X, y — no modification)
X[i][0] → weight values → scatter plot dots
        ↓
Step 4: Training
model.fit(X, y)
→ Decision Tree learns: "if weight > 120 → Orange, else → Mango"
        ↓
Step 4: Self-evaluation
model.predict(X)  → [1,1,1,1,1,1,1,0,0,0,0,0,0,0]
accuracy_score(y, predictions) → 1.0 → 100%
        ↓
Step 5: New data prediction
new_fruits = [[155,1],[92,0],[175,1],[88,0]]
model.predict(new_fruits) → [1, 0, 1, 0]
→ Orange, Mango, Orange, Mango
        ↓
Step 6: Visualisation of results
```

---

## 11. Why 100% Accuracy? Is That Normal?

### Short answer: No, 100% is NOT normal in real ML projects.

### Why we get 100% here:

**Reason 1 — Perfect data separation**
The mango range (80–110g) and orange range (130–200g) have a **20g gap** between them (110g to 130g). No overlap at all. The Decision Tree only needs one rule to classify everything correctly.

**Reason 2 — Testing on training data**
We call `model.predict(X)` where `X` is the same data we trained on. This is like letting a student mark their own exam using the answer sheet they already studied from. Of course they get 100%!

In real ML projects you always split data into **training set** and **test set**:
```python
# Real-world approach (not used here, to keep it simple for kids)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))
```

**Reason 3 — Dataset is tiny and artificial**
14 samples is extremely small. Real datasets have thousands to millions of samples with noisy, overlapping data.

### What overfitting means
If the mango and orange data overlapped (e.g., some mangoes weighed 145g and some oranges 105g), the Decision Tree might **memorise** the training data instead of **learning a general rule**. It would get 100% on training data but fail on new data. This is called **overfitting** — the Robot rote-learned without understanding!

---

## 12. What Could You Change? (Experiments)

These experiments help deepen understanding of how the model behaves:

### Experiment 1 — Add a confusing fruit
```python
# Add a mango that weighs 145g (overlaps with oranges!)
fruits_data.append([145, 0])
labels.append(0)
```
→ The accuracy will drop below 100% because no single weight threshold separates them.

### Experiment 2 — Remove the colour feature
```python
# Use only weight (drop column 1)
X_weight_only = X[:, 0:1]  # shape becomes (14, 1)
model.fit(X_weight_only, y)
```
→ Accuracy stays 100% because weight alone is sufficient for this dataset. But for a harder dataset, colour would matter.

### Experiment 3 — Add a 3rd fruit (multi-class)
```python
# Add grapes (ଅଙ୍ଗୁର) — very light, 20–40g, colour=2
fruits_data += [[25,2],[30,2],[35,2]]
labels      += [2, 2, 2]
```
→ Now it becomes a **3-class classification**. The Decision Tree handles this automatically!

### Experiment 4 — Change the model
```python
# Try a different algorithm
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X, y)
```
→ Compare accuracy with Decision Tree. On this dataset both should give 100%.

### Experiment 5 — Proper train/test split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model.fit(X_train, y_train)
print(f"Test accuracy: {accuracy_score(y_test, model.predict(X_test))*100:.0f}%")
```
→ More realistic evaluation — the model is tested on data it has *never seen*.

---

## Summary Table — Every Cell at a Glance

| Cell | Type | Purpose | Key Concept |
|------|------|---------|-------------|
| cell-0 | Markdown | Problem statement + story setup | Problem definition |
| cell-1 | Markdown | Step 1 intro | Analogies |
| cell-2 | Code | Import libraries | numpy, matplotlib, sklearn |
| cell-3 | Markdown | Step 2 intro + feature table | Features & labels |
| cell-4 | Code | Create dataset (X, y) | Dataset, arrays, labels |
| cell-5 | Markdown | Step 3 intro | EDA concept |
| cell-6 | Code | Draw scatter plot | Visualisation, separability |
| cell-7 | Markdown | Step 4 intro | Training analogy |
| cell-8 | Code | Train model + evaluate | `.fit()`, `.predict()`, accuracy |
| cell-9 | Markdown | Step 5 intro | Inference on new data |
| cell-10 | Code | Predict new fruits | `.predict()`, label_map, f-strings |
| cell-11 | Markdown | Step 6 intro | Victory celebration |
| cell-12 | Code | Draw pie + bar charts | subplots, pie, bar, axhline |
| cell-13 | Markdown | Glossary + next steps | Recap + future experiments |

---

*This notebook is intentionally kept simple so that a 10-year-old can run it and understand it. The ML concepts introduced here — data, features, labels, training, prediction, accuracy — are the same foundational ideas used in production ML systems at Google, Meta, and every AI company in the world. The scale is different, the concepts are identical.*

**ତୁ ଏବେ ଏକ AI Engineer! 🤖⭐**
