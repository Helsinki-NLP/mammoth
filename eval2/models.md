Naming conventions for tasks:
* sentmt: sentence-level MT with OPUS data (Tatoeba TC, HPLT, OpenSubtitles2024)
* docmt: document-level MT with synthetic data (max length 1024 characters)
* 4pivots: English/Spanish/German/French-centric MT training data (default otherwise is English-centric)
* denoise: monolingual denoising tasks (using doc-level synthetic data, English is original)
Naming conventions for parameter sharing:
* sharedenc: fully shared encoder (but still language-specific vocabs/embeddings)
* halfsharedenc: half of the encoder is shared across all languages, the other is language-specific
* LGAenc: encoders with three components: language-specific + shared across language group + fully-shared
* default is to have completely language-specific encoders and decoders
Naming conventions for model sizes:
* tiny: transformer-tiny (student) model with 2 decoder layers and model dimension = 256
* small: transformer-small (student) model with 2 decoder layers and model dimension = 512
* base (default): transformer-base model (6x6) and model dimension = 512
* big: transformer-big model (6x6) with 16 attention heads and double model dimensions (1024)
* xl: 12x12 transformer model with 16 attention heads and double model dimensions (1024)
## Models
Training on English/Spanish/German/French-centric machine translatino tasks
* docmt-4pivots
* docmt-4pivots-denoise-halfsharedenc-small
* docmt-4pivots-denoise-halfsharedenc-base
* docmt-4pivots-denoise-halfsharedenc-xl
Training on English-centric machine translation tasks + monolingual denoising:
* sentmt-denoise
* docmt-denoise
* docmt-denoise-sharedenc
* docmt-denoise-halfsharedenc
* docmt-denoise-LGAenc


