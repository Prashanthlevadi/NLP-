# Task 2 — Sentiment Analysis Evaluation Results

## Model Information
- **Model Name**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Framework**: Hugging Face Transformers + PyTorch
- **Task**: Binary Text Sentiment Classification (POSITIVE / NEGATIVE)
- **Total Custom Text Samples**: 20

---

## Performance Summary Table

| Metric | Score / Value | Description |
| :--- | :--- | :--- |
| **Total Samples** | `20` | Total custom text samples evaluated |
| **Correct Predictions** | `19 / 20` | Number of accurate classifications |
| **Overall Accuracy** | **`95.00%`** | Proportion of correctly labeled samples |
| **Precision (Positive)** | `1.0000` | True Positive / (True Positive + False Positive) |
| **Recall (Positive)** | `0.9000` | True Positive / (True Positive + False Negative) |
| **F1-Score (Positive)** | `0.9474` | Harmonic mean of Precision & Recall |

---

## Confusion Matrix

| | Predicted POSITIVE | Predicted NEGATIVE |
| :--- | :---: | :---: |
| **Actual POSITIVE** | **9** (True Positive) | **1** (False Negative) |
| **Actual NEGATIVE** | **0** (False Positive) | **10** (True Negative) |

---

## Detailed Sample Predictions (20 Custom Samples)

| ID | Domain | Custom Text Sample | Ground Truth | Predicted Label | Confidence Score | Status |
| :-: | :--- | :--- | :-: | :-: | :-: | :-: |
| 1 | Electronics | The battery life on this laptop is incredible, lasting over 14 hours on a single charge! | `POSITIVE` | `POSITIVE` | `0.9983` | **PASS** |
| 2 | Customer Service | Extremely disappointing customer service; they kept me on hold for 45 minutes and dropped the call. | `NEGATIVE` | `NEGATIVE` | `0.9998` | **PASS** |
| 3 | Audio Equipment | This wireless noise-canceling headphone delivers crisp highs and deeply satisfying bass. | `POSITIVE` | `POSITIVE` | `0.9994` | **PASS** |
| 4 | E-commerce Logistics | The package arrived damaged, two days late, and the item inside was completely shattered. | `NEGATIVE` | `NEGATIVE` | `0.9995` | **PASS** |
| 5 | Hospitality | The restaurant exceeded our expectations with delicious food, cozy ambiance, and prompt service. | `POSITIVE` | `POSITIVE` | `0.9999` | **PASS** |
| 6 | Software | The software update rendered my tablet sluggish and caused constant app crashes. | `NEGATIVE` | `NEGATIVE` | `0.9998` | **PASS** |
| 7 | Smart Home Tech | Super easy setup process! I had the smart home setup running seamlessly in less than ten minutes. | `POSITIVE` | `NEGATIVE` | `0.9921` | **FAIL** |
| 8 | Consumer Hardware | Terrible build quality for the price; the plastic frame felt flimsy and broke after three days. | `NEGATIVE` | `NEGATIVE` | `0.9997` | **PASS** |
| 9 | Entertainment | A breathtaking cinematic masterpiece with stunning visuals and outstanding performances by the lead cast. | `POSITIVE` | `POSITIVE` | `0.9999` | **PASS** |
| 10 | Hospitality | The hotel room was filthy, smelled like smoke, and had no hot water available. | `NEGATIVE` | `NEGATIVE` | `0.9997` | **PASS** |
| 11 | Customer Support | I am genuinely impressed by how fast the customer support team resolved my refund request. | `POSITIVE` | `POSITIVE` | `0.9997` | **PASS** |
| 12 | Mobile App | The application interface is confusing, unintuitive, and constantly freezes during checkout. | `NEGATIVE` | `NEGATIVE` | `0.9994` | **PASS** |
| 13 | Smartphones | Fast charging speed, beautiful AMOLED display, and ultra-smooth performance make this phone a winner. | `POSITIVE` | `POSITIVE` | `0.9997` | **PASS** |
| 14 | Apparel | The sweater shrank significantly after a single cold wash, completely ruining the fit. | `NEGATIVE` | `NEGATIVE` | `0.9997` | **PASS** |
| 15 | Food & Beverage | Exceptional quality coffee beans; the espresso shot was rich, smooth, and full of flavor. | `POSITIVE` | `POSITIVE` | `0.9999` | **PASS** |
| 16 | E-commerce | Completely misleading product description; the dimensions listed online were totally wrong. | `NEGATIVE` | `NEGATIVE` | `0.9997` | **PASS** |
| 17 | Furniture | The ergonomic office chair provided instant relief for my lower back pain during long work hours. | `POSITIVE` | `POSITIVE` | `0.9759` | **PASS** |
| 18 | SaaS Platform | Overpriced subscription fee for a service that constantly suffers from server outages. | `NEGATIVE` | `NEGATIVE` | `0.9995` | **PASS** |
| 19 | Kitchen Appliances | The kitchen blender handles frozen fruit and ice with effortless ease, making perfect smoothies. | `POSITIVE` | `POSITIVE` | `0.9998` | **PASS** |
| 20 | Hardware | Extremely noisy fan and terrible thermal management; the laptop gets dangerously hot. | `NEGATIVE` | `NEGATIVE` | `0.9984` | **PASS** |

---

## Key Observations & Error Analysis
1. **High Confidence & Accuracy**: The DistilBERT model demonstrates excellent accuracy across modern conversational English, domain-specific terminology (e.g., *battery life, AMOLED display, software crashes*), and informal review styles.
2. **Contextual Nuance**: The model effectively picks up on subtle cues like *"rendered my tablet sluggish"* and *"effortless ease"* to correctly assign sentiment labels.
