# Results

All numbers below come from `classification_report` output stored in `text_augmentation_evaluation.ipynb`. Class 3 is Science/Technology, the minority class in every imbalance scenario. Each cell shows Logistic Regression / SVM.

## Ceiling

On the original balanced dataset, both models reach 0.89 accuracy and 0.89 macro F1, with 0.86 F1 on class 3. This is the target that every balanced condition is measured against.

## Minority-class F1 (class 3)

| Scenario | Imbalanced | SMOTE | GPT-2 augmented |
|---|---|---|---|
| Severe under-sampling | 0.02 / 0.07 | 0.81 / 0.78 | 0.80 / 0.80 |
| Clustered minority instances | 0.26 / 0.25 | 0.85 / 0.85 | 0.84 / 0.84 |
| Topic-specific under-sampling | 0.49 / 0.45 | 0.85 / 0.85 | 0.85 / 0.85 |
| Progressive rarity | 0.00 / 0.04 | 0.77 / 0.75 | 0.57 / 0.69 |

## Minority-class precision and recall (class 3, Logistic Regression)

| Scenario | Imbalanced P / R | SMOTE P / R | Augmented P / R |
|---|---|---|---|
| Severe under-sampling | 0.24 / 0.01 | 0.87 / 0.76 | 0.86 / 0.75 |
| Clustered minority instances | 0.35 / 0.21 | 0.87 / 0.84 | 0.81 / 0.87 |
| Topic-specific under-sampling | 0.43 / 0.58 | 0.83 / 0.88 | 0.86 / 0.84 |
| Progressive rarity | 0.27 / 0.00 | 0.86 / 0.70 | 0.92 / 0.41 |

## Macro F1

| Scenario | Imbalanced | SMOTE | GPT-2 augmented |
|---|---|---|---|
| Severe under-sampling | 0.21 / 0.22 | 0.86 / 0.85 | 0.85 / 0.85 |
| Clustered minority instances | 0.34 / 0.32 | 0.89 / 0.88 | 0.88 / 0.88 |
| Topic-specific under-sampling | 0.45 / 0.42 | 0.88 / 0.87 | 0.88 / 0.87 |
| Progressive rarity | 0.19 / 0.20 | 0.82 / 0.80 | 0.72 / 0.77 |

## What the numbers show

Imbalance is catastrophic, and both balancing strategies fix most of it. Under severe under-sampling the classifier essentially stops predicting the minority class at all, with recall at 0.01. Under progressive rarity it stops entirely, with F1 at 0.00. Both SMOTE and GPT-2 augmentation recover that to the 0.75 to 0.85 range, close to the balanced ceiling of 0.86.

GPT-2 augmentation did not outperform SMOTE. It matches SMOTE within one point on severe under-sampling, clustered minority instances, and topic-specific under-sampling. On progressive rarity, the most extreme imbalance, it loses clearly: 0.69 F1 at best against SMOTE's 0.77.

The most informative result is the shape of that loss. On progressive rarity the augmented models produce the highest minority-class precision anywhere in the experiment, 0.92 for Logistic Regression, while recall falls to 0.41. The synthetic text was consistent enough to sharpen the decision boundary but too generic to broaden it, so the model became confident and narrow. SMOTE, interpolating between real vectors, kept more of the minority class in range.

The one place augmentation edges ahead is clustered minority instances, where it reaches 0.87 recall on class 3 against SMOTE's 0.84. That scenario preserves text-length diversity in the minority class, which gives the generator more varied material to continue from.

## Why the result came out this way

Three implementation choices explain most of it, and all three are fixable:

- Generation used base GPT-2 with no prompt conditioning. Each synthetic sample is a continuation of an existing minority-class text, capped at 50 new tokens, so it mostly restates what the model already had rather than adding new regions of the class.
- The generated text is fed through the same TF-IDF vectorizer capped at 5,000 features. Whatever contextual information the generator contributed is discarded at the feature layer, which is the step where a generative approach should have had its advantage.
- No quality filter was applied. Nothing verifies that a generated sample belongs to the class it was generated for, and nothing removes near-duplicates of existing training examples.

A fair test of the hypothesis would keep the imbalance scenarios and swap all three: an instruction-tuned generator with scenario-conditioned prompts, a transformer classifier instead of TF-IDF, and a filtering stage between generation and training.

## Reproducibility note

The plotting cells near the end of the notebook read from a stored results dictionary and apply an adjustment step before rendering the heatmap and radar chart. The tables in this document are taken directly from the model runs.
