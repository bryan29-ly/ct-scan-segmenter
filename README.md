# 🩻 ScanAssist — AI Pre-Screening for CT Scans

Automatic segmentation of **54 anatomical structures** from abdominal CT slices, with **Mistral AI-powered radiology summaries** in plain language.

> Built for the [Alan × Mistral AI Health Hack 2025](https://lu.ma/).

![demo](assets/demo.gif) <!-- Replace with your screen recording -->

---

## The problem

Radiologists spend significant time manually identifying and measuring organs on CT scans. Patients receive reports full of jargon they don't understand. In high-volume imaging centers, triage bottlenecks delay care.

## What ScanAssist does

1. **Segments** 54 anatomical structures (organs, vessels, bones, tumors) from a single CT slice in < 1 second
2. **Analyzes** the segmentation: structure count, coverage, relative sizes, dominant structures
3. **Generates** a plain-language pre-screening report via Mistral AI — readable by both clinicians and patients

The pipeline: `CT Slice → U-Net Segmentation → Structure Analysis → Mistral Report`

---

## From competition to product

### The competition

The segmentation model at the core of ScanAssist was developed during the **[ENS × Raidium Data Challenge](https://challengedata.ens.fr/)**, a national data science competition organized by ENS Paris and Raidium, focused on medical image segmentation. The model **reached the top 5** of the competition leaderboard.

The full training code, experiments, and methodology are available in the dedicated repository:
👉 **[ens-data-challenge](https://github.com/bryan29-ly/ENSxRaidium-Data-Challenge)** — training pipeline, loss functions, data processing, and evaluation.

### Why this competition was interesting

The dataset presents three compounding constraints that make standard approaches fail:

**Partial annotations.** Each training scan is only labeled for a *subset* of organs. The liver might be annotated on scan A but unlabeled on scan B — even though it's visible. A naive model treats missing labels as "background," learning the wrong thing.

**Severe class imbalance.** 54 structures ranging from large organs (thousands of pixels) to tiny vessels (dozens of pixels). Standard losses let dominant classes crush rare ones.

**Limited labeled data.** Only 800 partially-annotated images + 1200 unlabeled scans. Standard supervised training drastically overfits.

### How we solved it

| Constraint | Solution |
|---|---|
| Partial annotations | **Marginal segmentation loss** — masks out unannotated classes so the model only learns from verified labels |
| Class imbalance | **Inverse-square-root frequency sampling** — upweights images containing rare structures |
| Limited data | Heavy spatial + intensity augmentations via Albumentations |

**Architecture:** 7-stage PlainConv U-Net with deep supervision (3 heads at different resolutions), InstanceNorm, LeakyReLU. Trained with SGD + polynomial LR decay for 500 epochs.

### From leaderboard to real-world use case

A model that ranks well in a competition isn't a product. **ScanAssist is an exercise in taking a high-performing research model and putting it in the hands of a real user** — turning raw segmentation masks into something a clinician or a patient can understand in seconds. This meant building a clean interface, extracting meaningful statistics from the output, and adding a natural-language report layer via Mistral to bridge the gap between pixels and clinical insight.

---

## Quick start

```bash
git clone https://github.com/bryan29-ly/ct-scan-segmenter.git
cd ct-scan-segmenter
pip install -r requirements.txt
streamlit run app.py
```

Model weights (~280 MB) download automatically from [Hugging Face](https://huggingface.co/bryan29-ly/ct-scan-segmenter) on first launch.

### Mistral AI report (optional)

Get a free API key at [console.mistral.ai](https://console.mistral.ai/) and paste it in the sidebar. Without a key, the app shows an example report.

---

## Project structure

```
├── app.py             # Streamlit interface
├── model.py           # PlainConv U-Net architecture
├── inference.py       # Weight loading + preprocessing + prediction
├── report.py          # Mistral API integration
├── examples/          # Sample CT slices (.npy, 256×256)
├── requirements.txt
└── README.md
```

---

## What's next — scaling ScanAssist

This demo is a proof of concept. Here's how it could evolve into a real clinical tool:

**Semantic labels.** The current model outputs numeric structure IDs (1–54) without organ names. Training on a larger, fully-annotated dataset (e.g. the full TotalSegmentator with 104 labeled classes) would give the model semantic understanding — identifying organs by name, not just by cluster ID.

**Smarter reports via RAG.** Instead of generating reports from pixel statistics alone, the Mistral integration could query a medical knowledge base (clinical guidelines, anatomy references, pathology databases) through retrieval-augmented generation. This would produce reports grounded in medical literature — flagging size deviations from population norms, or suggesting differential diagnoses based on organ location and morphology.

**3D volume analysis.** Extending from single 2D slices to full 3D CT volumes would enable volumetric organ measurements, which are far more clinically relevant.

**Health platform integration.** A tool like this could feed into platforms like [Alan](https://alan.com/) — connecting automated scan analysis with the patient's health record, prevention recommendations, and care coordination.

---

## Data & credits

- **CT images** from the [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) dataset (Wasserthal et al., 2023), provided through the ENS × Raidium challenge
- **Challenge** organized by [ENS Paris](https://www.ens.fr/) and [Raidium](https://raidium.eu/)
- **AI reports** powered by [Mistral AI](https://mistral.ai/)

## License

Code is MIT-licensed. Model weights are released for research and demonstration purposes only — **not for clinical use**.
