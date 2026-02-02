Mammoth now supports pretokenization (automatic/manual) before training starts:

- Add `data_type: indexed` to enable pretokenization. For each text file, pretokenization will produce a `idx` file and a `bin` file.
- Pretokenization is incompatible with on-the-fly transforms.
- Currently, length filtering is not included in the pretokenization process. Please keep the length of the training data less than or equal to `max_length: 512` otherwise training will crash. A template script for using HF tokenizer for length filtering is provided below.
- To use automatic pretokenization, assign the paths with the filtered text data. Mammoth will first check if the provided text data has been pretokenized (looking for `idx` and `bin` files in the same directory). If not, pretokenization will run automatically and process this data for training, saving the results to the same directory.
- To use manual pretokenization, run the script `mammoth/scripts/preprocess_indexed.py`. This is a template script:

```bash
python mammoth/scripts/preprocess_indexed.py \
  --input /scratch/project_462000964/members/wangchao/training/test_pretokenization/train_ar_filtered.txt \
  --output_prefix train_ar_tok \
  --vocab_path hf_tokenizer.json \
  --workers 8
```

During training, provide the output prefix `train_ar_tok` as the data paths for Mammoth to load the correct input data.

Configuration example:

```yaml
task_ar-eng:
  src_tgt: "ara-eng"
  data_type: indexed # Enables automatic pretokenization/training with pretokenized data 
  weight: 1.0
  introduce_at_training_step: 0
  node_gpu: "0:0"
  enc_sharing_group: ["ara","all"]
  dec_sharing_group: ["all","eng"]
  # transforms: [denoising,filtertoolong] # Do not use any transforms here
  
  # When using automatic pretokenization, provide the paths to the text files:
  path_src: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/train_ar_filtered.txt
  path_tgt: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/train_en_filtered.txt
  path_valid_src: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/val_ar_filtered.txt
  path_valid_tgt: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/val_en_filtered.txt
  
  # When using manual pretokenization, provide the output prefix (Mammoth will locate the .bin and .idx files):
  path_src: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/train_ar_tok # the `output_prefix` value
  path_tgt: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/train_en_tok # the `output_prefix` value
  path_valid_src: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/val_ar_tok # the `output_prefix` value
  path_valid_tgt: /scratch/project_462000964/shared/hplt_bilingual/ar-en.tmx/val_en_tok # the `output_prefix` value
  
 task_fi_fi: 
  src_tgt: "fin-fin"
  weight: 1.0
  introduce_at_training_step: 0
  node_gpu: "0:1"
  enc_sharing_group: ["fin","all"]
  dec_sharing_group: ["all","fin"]
  transforms: [denoising,filtertoolong] # Transforms work only with non-indexed datasets 
  path_src: /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/train_fi_filtered.txt
  path_tgt: /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/train_fi_filtered.txt
  path_valid_src: /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/val_fi_filtered.txt
  path_valid_tgt: /scratch/project_462000964/shared/hplt_bilingual/fi-en.tmx/val_fi_filtered.txt
 ...

src_seq_length_max: 512 
tgt_seq_length_max: 512 
src_seq_length_min: 10 
tgt_seq_length_min: 10 
max_length: 512 # Input sentence length must be less than or equal to this value when `filtertoolong` transform is not used!
```

The template script for using HF tokenizer for length filtering:

```bash
python mammoth/scripts/filter_by_length.py \
  --src_input ${SRC_INPUT} \
  --tgt_input ${TGT_INPUT} \
  --src_output ${SRC_OUTPUT} \
  --tgt_output ${TGT_OUTPUT} \
  --src_tokenizer ${SRC_TOKENIZER} \
  --tgt_tokenizer ${TGT_TOKENIZER} \
  --min_length 10 \
  --max_length 512
```

For separate source and target tokenizers, use `--src_tokenizer` and `--tgt_tokenizer` instead of `--tokenizer`.