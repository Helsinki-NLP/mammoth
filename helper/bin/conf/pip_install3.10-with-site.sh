#! /usr/bin/bash
# pip_install.sh -- builds a mammoth pip environment

exit 1

# Download
if [ ! -d  $PROJHOME/rebuilt3.10-with-conda ]; then
    mkdir -p $PROJHOME/rebuilt3.10-with-conda
fi
if [ ! -d  $PROJHOME/rebuilt3.10-with-conda/mammoth ]; then
  cd       $PROJHOME/rebuilt3.10-with-conda
  git clone https://github.com/Helsinki-NLP/mammoth.git
fi
export PYTHONUSERBASE="$PROJHOME/venv/mammoth3.10-with-conda"
export CODE_DIR="$PROJHOME/rebuilt3.10-with-conda/mammoth"

source init3.10-with-conda.sh

singularity exec $SIF bash <<'EOF'

  echo "Running inside a container!"
  echo "\$WITH_CONDA"
  $WITH_CONDA

  if [ ! -d "$PROJHOME/venv/mammoth3.10-with-conda" ]; then
      echo Creating virtual environment mammoth3.10-with-site
      echo python -m venv "\$PROJHOME/venv/mammoth3.10-with-site"
      python -m venv --system-site-packages "$PROJHOME/venv/mammoth3.10-with-site" 
  else
      echo Found a virtual environment mammoth3.10-with-site
  fi
  echo Activating ...
  source $PROJHOME/venv/mammoth3.10-with-site/bin/activate
  python --version  
 
    pip install --upgrade pip setuptools wheel
    pip install Flask==2.0.3
    pip install configargparse
    pip install flake8==4.0.1
    pip install pytest-flake8==1.1.1
    pip install pytest==7.0.1
    pip install pyyaml==6.0.2
    pip install pyonmttok
    pip install sacrebleu==2.3.1
    pip install scikit-learn==1.3.1  
    pip install sentencepiece==0.2.0
    pip install tensorboard==2.19.0
    pip install timeout-decorator
    pip install tqdm==4.67.1
    pip install waitress
    pip install x-transformers==1.32.14

    cd $CODE_DIR
    echo $CODE_DIR
#    pip install -e .

  deactivate

EOF


