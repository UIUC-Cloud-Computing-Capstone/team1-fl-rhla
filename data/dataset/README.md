# Data

Datasets used in HRALoRA. Paths and splits are set in the experiment configs (e.g. under `config/` and experiment YAMLs).

## CIFAR-100

- **Description:** 100-class image classification; 60k 32×32 colour images (50k train, 10k test).
- **Source:** [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) (auto-downloaded by the framework).
- **Usage:** Vision experiments; ViT + LoRA fine-tuning.
- **Citation:** Alex Krizhevsky and Geoffrey Hinton, “Learning multiple layers of features from tiny images,” 2009.

## LEDGAR

- **Description:** Large-scale multi-label corpus for classifying legal provisions in contracts.
- **Source:** [LEDGAR on HuggingFace](https://huggingface.co/datasets/ledgar) or LREC 2020.
- **Usage:** NLP experiments; BERT + LoRA fine-tuning.
- **Citation:** Don Tuggener, Pius Von Däniken, Thomas Peetz, and Mark Cieliebak, “Ledgar: a large-scale multi-label corpus for text classification of legal provisions in contracts,” in *International Conference on Language Resources and Evaluation (LREC)*, 2020.
