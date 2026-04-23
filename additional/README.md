## Objective

The main task was to implement an end-to-end fine-tuning pipeline for an Automatic Speech Translation based on the openai/whisper-base architecture to perform direct speech translation from English audio to Ukrainian text. The quality of the generated translations is evaluated using the COMET metric

## Technical Implementation

- Base Model: openai/whisper-base (approximately 245 million parameters)

- Optimization: Parameter-Efficient Fine-Tuning (PEFT) using LoRA. LoRA was configured with r=32, lora_alpha=64, and targeted the attention modules (q_proj, v_proj). This allowed for training only 3,538,944 parameters (1.44% of the total network) while keeping the base model frozen 

- Dataset: The google/fleurs dataset was used, aligning the en_us audio splits with uk_ua text transcriptions. A subset of 2,500 aligned records was used for training

- Audio Processing: Audio was resampled to 16 kHz and converted into log-Mel spectrograms via WhisperProcessor

## Results

The model was trained for 5 epochs using gradient clipping to maintain stability. To avoid hallucinations during inference, generation parameters were adapted during testing (num_beams=5, repetition_penalty=1.5, no_repeat_ngram_size=3, max_new_tokens=100)

Final COMET System Score: 0.4690

    Генерація (Укр): після 24 вересня 1759 артур гіннес признався 9000-річним лісом на речі сент-жеймської дворі в дубліній ірландії

    Еталон (Укр):    24 вересня 1759 артур гіннесс підписав контракт на 9000-річну аренду пивоварні в районі воріт святого джеймса в дубліні ірландія


## Hypotheses, insights

During the development of the pipeline, several critical architectural constraints and deep-learning behaviors were observed:

1. Initial attempts to use Whisper's native task="translate" token failed because the model is pre-trained to translate audio exclusively into English. Passing language="uk" alongside the translate task caused a weight conflict. The pipeline was restructured to use task="transcribe" with the target language explicitly set to Ukrainian. This forced the model to bypass its Any-to-English bias and construct a direct English audio to Ukrainian text mapping  

2. Attempting full fine-tuning on a small dataset previously led to NaN losses (exploding gradients) and complete degradation of the model's native Ukrainian vocabulary. By implementing LoRA, 98.5% of the model's weights were frozen. This completely eliminated the NaN loss issue and preserved the base model's grammatical capabilities, increasing the final COMET score to 0.4690

3. The inference logs reveal pretty normal cross-lingual phonetic mappings. Because the model lacks the massive bilingual text corpus required for deep semantic translation, it often resorts to transliterating English sounds into Ukrainian words. For example, "Former U.S. speaker" sounded like "yogurt" to the model, generating "йогурт-спікор". Similarly, "9000-year lease" was translated using the Ukrainian word "лісом" due to phonetic similarity. This proves the AST feature extractor successfully bridged the audio-text gap, though semantic mapping remains constrained by the limited dataset size (2,500 samples)

## Conclusion

The model successfully generates structured Ukrainian text directly from English audio, achieving a COMET score of 0.4690. Possible approach to achieve SOTA-level semantics would require significantly expanding the parallel audio-text dataset and therefore involve more computational power