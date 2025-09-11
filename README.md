# 🦣 MAMMOTH: Massively Multilingual Modular Open Translation @ Helsinki
This repository contains the code for 🦣 MAMMOTH, the modular translation toolkit from Helsinki-NLP.

This library is built on top of OpenNMT-py.
[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) is the [PyTorch](https://github.com/pytorch/pytorch) version of the [OpenNMT](https://opennmt.net) project, an open-source (MIT) neural machine translation framework. It is designed to be research friendly to try out new ideas in translation, summary, morphology, and many other domains. Some companies have proven the code to be production ready.

### Documentation
The original ONMT-py documentation is available [here](https://opennmt.net/OpenNMT-py/).  
Our own modifications (currently-under-development) are documented [here](https://helsinki-nlp.github.io/mammoth/).

FINAL NOTE: We would appreciate it A LOT if you report issues with this repo and/or the documentation 

### Acknowledgements
We thank the NVIDIA AI Technology Center Finland for their help with the multi-gpu/node implementation.

### Quick Installation:

1. Assume that you are a part of project 2005099 (CSC) or 462000964 (LUMI).  You may be part of some other project with similar numbers.

2. Add the following to your `.profile` and login again to the machine.
```
# Find the system name and account
HOSTNAME=$(hostname)  # this does not work well for lumi
if [[ $SYSTEM == "" ]]; then
        if [[ "$HOSTNAME" == roihu* ]]; then
            export SYSTEM=roihu
            export ACCOUNT=project_2005099
        elif [[ "$HOSTNAME" == mahti* ]]; then
            export SYSTEM=mahti
            export ACCOUNT=project_2005099
        elif [[ "$HOSTNAME" == puhti* ]]; then
            export SYSTEM=puhti
            export ACCOUNT=project_2005099
        elif grep -q 'lumi-super' /etc/motd; then
            export SYSTEM=lumi
            export ACCOUNT=project_462000964
        else
            echo "⚠️ Unknown system: $HOSTNAME. Please edit this script (detect_system.sh) manually."
            exit 1
        fi
fi
export PROJHOME=/project/$ACCOUNT/members/$USER
export PROJDATA=/scratch/$ACCOUNT/members/$USER
export GITHOME=$PROJHOME/git
echo "SYSTEM=$SYSTEM ACCOUNT=$ACCOUNT"
echo "PROJHOME=$PROJHOME"
echo "PROJDATA=$PROJDATA"
echo "GITHOME=$GITHOME"
```

3. Now create the directories and clone the MAMMOTH and check out the feat/helper branch:
```
mkdir -p $PROJHOME $GITHOME
cd $GITHOME
git clone git@github.com:Helsinki-NLP/mammoth.git
git worktree add ../mammoth-helper feat/helper
```

4. Start the building script:
$GITHOME/mammoth-helper/helper/bin/conf/build-venv-mammoth-hf.sh

You should see:
```
Successfully installed certifi-2025.8.3 charset-normalizer-3.4.3 configargparse-1.7.1 einx-0.3.0 flake8-4.0.1 flask-2.0.3 frozendict-2.4.6 idna-3.10 loguru-0.7.3 mammoth-nlp-0.2.1 markupsafe-3.0.2 mccabe-0.6.1 networkx-3.5 protobuf-6.32.0 pycodestyle-2.8.0 pyflakes-2.4.0 pytest-flake8-1.1.1 sympy-1.14.0 transformers-4.55.4 typing_extensions-4.14.1 urllib3-2.5.0 waitress-3.0.2 x-transformers-1.32.14
[build-venv-mammoth-hf] installed mammoth and the requirements
```

5. Check that you $PROJHOME now looks like:
```
├── bin -> helper/bin
├── git
│   ├── mammoth
│   ├── mammoth-helper
│   └── mammoth-hf
├── helper -> /project/$ACCOUNT/members/$USER/git/mammoth-helper/helper
├── images -> helper/images
├── lib -> helper/lib
├── mammoth -> /project/$ACCOUNT/members/$USER/git/mammoth-hf
└── venv
    └── mammoth-hf
```

6. You can use the mammoth-hf (feat/hf-integration) branch with the virtual
environment `venv/mammoth-hf`.  There are more useful scripts in `bin`, but
some of them may need more debugging.  More information will follow.


