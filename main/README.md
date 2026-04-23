# Report (Main Task)

## Data Overview, Main Features
The task utilizes the Toronto Dataset, consisting of Ukrainian audio recordings and corresponding text transcripts. 
* The inputs are resampled to $16$ kHz sampling rate required by the Whisper architecture.
* Provided dataset consists of ~$18300$ audio files. Around $11000$ files mentioned in the `labels.jsonl` are missing.

## Description of Approaches Used
* **Model Architecture:** The system fine-tunes the `openai/whisper-small` pre-trained model. Whisper is a transformer-based sequence-to-sequence model. The generation config is (`language="uk"`, `task="transcribe"`).
* **Precision:** Training utilizes mixed-precision training (`16-mixed`)
* **Data Handling:** A list defines the test set, while the remaining files are split into train (90%) and validation (10%) sets. `DataCollator` pads audio features and text tokens to the longest sequence in the batch.
* **Optimization:** Training utilizes the AdamW optimizer with a Cosine Annealing lr scheduler over 10 epochs. Gradient clipping (`gradient_clip_val=1.0`) is applied too.

## Metrics Tables
The model is evaluated using two metrics:
* **WER (Word Error Rate):** Measures the percentage of words that were incorrectly predicted (insertions, deletions, substitutions).
* **CER (Character Error Rate):** Measures the percentage of incorrect characters.

| Word Error Rate (WER) | Character Error Rate (CER) |
| --------------------- | -------------------------- |
| 0.2565 | 0.1107 |

## Hypotheses
* **Hypothesis on Text Normalization:** Stripping punctuation and lowercasing during the `normalize_text` evaluation step will result in significantly better WER/CER scores than raw inference, as the model won't be penalized for missing commas or capitalization nuances.
* **Hypothesis on Model Capacity:** `whisper-small` provides a good balance between training speed and accuracy. However, because it is fine-tuned without audio augmentations, it may overfit to the specific acoustic conditions of the Toronto dataset.