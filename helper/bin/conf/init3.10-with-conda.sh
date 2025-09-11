# source this file

# module load LUMI/23.09  partition/L
#   LUMI/23.09: Though before the summer-of-2024 system update we expected to be capable to
#   support this LUMI stack longer, the choice for ROCm 6.0 rather than 5.7 may
#   cause problems that we cannot fix and may require updating to 24.03.

module load LUMI/24.03  partition/L
module use /appl/local/containers/ai-modules
module load singularity-AI-bindings                # AI bindings will be needed

export BINDINGS="--bind /bin/ip:/bin/ip --bind /usr/lib64/libmnl.so.0:/usr/lib64/libmnl.so.0 \
  --bind /opt/cray/libfabric/1.15.2.0/bin/fi_info:/bin/fi_info "
export SIF=$PROJHOME/sif/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif 

echo "Use the following commands to enter the env:"
echo "  source init3.10-with-conda.sh"
echo "  singularity exec \$BINDINGS \$SIF bash"
echo "  \$WITH_CONDA"
echo "  source \$PROJHOME/venv/mammoth3.10-with-conda/bin/activate"
echo "Then exit the container with"
echo "  deactivate"
echo "  exit"

