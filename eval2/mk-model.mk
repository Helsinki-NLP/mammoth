# -----------------------------------------------------------------------------
# Model aliases
# -----------------------------------------------------------------------------
# Shared base directory for model aliases.
MODELS := /scratch/project_462000964/members/tiedeman/MARMoT/models
SODELS := /scratch/project_462000964/members/tiedeman/MARMoT/sandbox/tiedeman/oellm-lg

# Add model aliases here.
MODEL_docmt4denhalfbig    := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-big/mammoth
MODEL_docmt4denhalfbase   := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-base/mammoth
MODEL_docmt4denhalfsmall  := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-small/mammoth
MODEL_docmt4denhalfxl     := $(MODELS)/docmt-4pivots-denoise-halfsharedenc-xl/mammoth
MODEL_docmt10denhalf      := $(MODELS)/docmt-10pivots-denoise-halfsharedenc/mammoth
#MODEL_docmt4densmall      := $(MODELS)/docmt-4pivots-denoise-small/mammoth
MODEL_docmt4              := $(MODELS)/docmt-4pivots/mammoth
MODEL_finnish             := $(MODELS)/finnish/mammoth
MODEL_finnish_denoise_xl  := $(MODELS)/finnish-denoise-xl/mammoth
MODEL_denoise             := $(MODELS)/docmt-denoise/mammoth
MODEL_d_fincentric        := $(MODELS)/docmt-denoise-fincentric/mammoth
MODEL_d_halfsharedenc     := $(MODELS)/docmt-denoise-halfsharedenc/mammoth
MODEL_d_sharedenc         := $(MODELS)/docmt-denoise-sharedenc/mammoth
MODEL_LGAenc              := $(MODELS)/docmt-denoise-LGAenc/mammoth
MODEL_LGAenc_fincentric   := $(MODELS)/docmt-denoise-LGAenc-fincentric/mammoth
MODEL_sentmtdenoise       := $(MODELS)/sentmt-denoise/mammoth
MODEL_sentmthalfsmall     := $(MODELS)/sentmt-halfsharedenc-small/mammoth
MODEL_sentmthalfxl        := $(MODELS)/sentmt-halfsharedenc-xl/mammoth
MODEL_ALIASES := docmt4denhalfbase docmt4denhalfsmall docmt4denhalfxl \
  docmt10denhalf docmt4 finnish finnish_denoise_xl denoise d_fincentric \
  d_halfsharedenc d_sharedenc LGAenc LGAenc_fincentric sentmtdenoise sentmthalfsmall \
  sentmthalfxl
