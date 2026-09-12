# Text Augmentation using LLMs

Improving text classification on under-represented categories by generating synthetic training text with a language model, and comparing that approach against traditional resampling.

## Authors

- Sejal Agarwal, UMass Amherst
- Siddharth Jain, UMass Amherst

## What this repository contains

- `text_augmentation_evaluation.ipynb`: the full experiment, from dataset loading through evaluation.
- `TextAugmentationUsingLLM.pdf`: the write-up covering motivation, method, and results.

## The problem

Text classifiers trained on imbalanced data learn to favor the majority classes and quietly fail on the minority one. Overall accuracy stays high while minority-class recall collapses, so the metric most people look at hides the failure.

Resampling methods like SMOTE were designed for tabular features. Applied to TF-IDF vectors, they interpolate between sparse vectors rather than producing anything that reads as real text. The question here is whether generating new text with a language model produces a better minority class than interpolating existing vectors does.

## Dataset

AG News, loaded from Hugging Face. Four balanced classes:

| Label | Class |
|---|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Science/Technology |

120,000 training samples and 7,600 test samples in the original distribution. Imbalance is introduced deliberately, so the balanced version acts as the performance ceiling.

## Imbalance scenarios

Rather than testing a single flat under-sampling, the notebook builds four different shapes of imbalance, because they fail in different ways:

1. Severe under-sampling across multiple classes. Several classes reduced to a small fraction of their original size.
2. Clustered minority instances with diverse text lengths. The minority class is kept but restricted to a narrow cluster, so the classifier sees few examples and little variety.
3. Topic-specific under-sampling within a category. An LDA topic model is fit on one class, and instances matching a chosen topic are removed. This simulates a gap inside a class rather than a shortage of the class overall.
4. Progressive rarity. A long-tail distribution where class frequency drops off gradually instead of in one step.

## Pipeline

Preprocessing is held constant across every condition so that differences in results come from the balancing strategy and nothing else.

- Cleaning: lowercase, digits removed, punctuation removed, whitespace normalized, duplicates and nulls dropped.
- Features: TF-IDF, capped at 5,000 features.
- Models: Logistic Regression (saga solver) and LinearSVC.
- Evaluation: per-class precision, recall, and F1 from `classification_report`, with attention on the minority class rather than overall accuracy.

## Conditions compared

Each imbalance scenario is run through three conditions:

- Imbalanced baseline, untouched. This is the floor.
- SMOTE oversampling applied to the TF-IDF vectors.
- Generative augmentation, where synthetic text is generated for the minority class using a Hugging Face `text-generation` pipeline (GPT-2 in this notebook), then appended to the training data before vectorization.

## Running it

The notebook was written for Google Colab with a T4 GPU and expects that environment.

1. Open the notebook in Colab and set the runtime to GPU.
2. Mount Google Drive. Trained models are cached to a Drive path so runs can be resumed without retraining, so update `model_save_dir` to a path you own.
3. Run cells in order. Dataset download, preprocessing, and baseline training come first, then the four imbalance constructions, then SMOTE, then generative augmentation, then evaluation.

A Hugging Face login cell is included for gated model access. It is not required for the GPT-2 path.

## Results

Both balancing strategies recover most of what imbalance destroys. Minority-class F1 falls as low as 0.02 on the imbalanced baseline and returns to the 0.75 to 0.85 range after balancing, against a 0.86 ceiling from the original balanced dataset.

GPT-2 augmentation did not outperform SMOTE. It matches SMOTE within a point on three of the four imbalance scenarios and underperforms on progressive rarity. The useful finding is the shape of that loss: under the most extreme imbalance, augmentation produces the highest minority-class precision in the experiment, 0.92, while recall drops to 0.41. The synthetic text was coherent enough to sharpen the decision boundary but too generic to widen it.

Full per-scenario tables, the precision-recall breakdown, and an analysis of why the result came out this way are in [RESULTS.md](RESULTS.md).


## Known limitations

- AG News is widely available and likely appeared in the generator's pretraining data, so some of the gain may reflect memorization of the domain rather than useful augmentation. Testing on a private or domain-specific corpus would separate the two.
- Generation length was capped for runtime, which produces short synthetic samples that carry less signal than real ones.
- There is no quality gate on generated text. Nothing verifies that a synthetic sample actually belongs to the class it was generated for, and nothing checks for near-duplicates of existing training examples.
- TF-IDF discards word order and context, which throws away much of what a generative model contributes. The representation is the bottleneck.

## Future work

- Replace TF-IDF and classical models with a fine-tuned transformer classifier such as RoBERTa, so contextual information in the generated text survives into the features.
- Add a filtering stage: classify each generated sample before accepting it, deduplicate against the originals, and check for distribution drift.
- Use a larger instruction-tuned generator with prompts conditioned on the specific gap in each imbalance scenario, and measure how much of the improvement comes from model size versus prompt design.
- Validate on a real imbalanced dataset from a domain where the minority class matters, such as fraud detection or clinical text.

## License

MIT. See [LICENSE](LICENSE).
