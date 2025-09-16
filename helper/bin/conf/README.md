# Installing and setting up the MAMMOTH with helper features

The new helper features of MAMMOTH have been designed to facilitate installation and
correct running, and monitoring of the software using Slurm or Torchrun in all CSC-maintained
supercomputers: puhti.csc.fi, mahti.csc.fi, lumi.csc.fi, (and roihu.csc.fi).  This README describes the installation procedure.

1. You need to be a part of project that allows to use CSC super
   computers.  Once you know your project and related computers, you
   can following these instructions.  For example, assume that you are
   a part of the projects 2005099 (CSC) or 462000964 (LUMI).  If you
   are not part of these projects, but you belong to some other
   project, you will have similar numbers for them.

   Add the following to your `.profile` and login again to the machine.
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
        elif egrep -q 'lumi|LUMI' /etc/motd; then
            export SYSTEM=lumi
            export ACCOUNT=project_462000964
        else
            export SYSTEM=unknown
            export ACCOUNT=project_462000964
            echo "⚠️ Unknown system: $HOSTNAME."
            egrep 'lumi-super|LUMI' /etc/motd
        fi
   fi
   export PROJHOME=/project/$ACCOUNT/members/$USER
   export PROJDATA=/scratch/$ACCOUNT/members/$USER
   export SHARDATA=/scratch/$ACCOUNT/shared
   export GITHOME=$PROJHOME/git
   export SBATCH_ACCOUNT="$ACCOUNT # sbatch honors SBATCH_* env vars as defaults
   ```

2. Make the directories and link them
   ```
   source ~/.profile
   mkdir -p $PROJDATA $PROJHOME $GITHOME
   ln -s $PROJDATA $PROJHOME/data
   ```  

3. Clone the main branch and check out the helper branch -- (they are not yet merged)
   ```
   cd  $GITHOME
   git clone https://github.com/Helsinki-NLP/mammoth.git mammoth
   cd mammoth
   git fetch origin feat/helper
   git worktree add --track -b feat/helper ../mammoth-helper origin/feat/helper
   ```
4. Build the inheritable `base` directory `$PROJHOME/base` (that extends your venv)
   ```
   $GITHOME/mammoth-helper/helper/bin/conf/build-venv.sh
   ```
   Running this takes a couple of minutes. Now you should see:
   ```
   Successfully installed certifi-2025.8.3 charset-normalizer-3.4.3 configargparse-1.7.1 einx-0.3.0 flake8-4.0.1 flask-2.0.3 frozendict-2.4.6 idna-3.10 loguru-0.7.3 mammoth-nlp-0.2.1 markupsafe-3.0.2 mccabe-0.6.1 networkx-3.5 protobuf-6.32.0 pycodestyle-2.8.0 pyflakes-2.4.0 pytest-flake8-1.1.1 sympy-1.14.0 transformers-4.55.4 typing_extensions-4.14.1 urllib3-2.5.0 waitress-3.0.2 x-transformers-1.32.14
   [build-venv-mammoth-hf] installed mammoth and the requirements
   ```

   This means that MAMMOTH and its requirements have been installed,
   and - importantly - you also have helper features.

5. The directory structure of your project home will look like this:

```
.
├── base
│   ├── git -> /project/project_462000964/members/aylijyra/git
│   ├── mammoth-helper -> git/mammoth-helper
│   ├── mammoth-hf -> git/mammoth-hf
│   └── venv -> /project/project_462000964/members/aylijyra/venv
├── data -> /scratch/project_462000964/members/aylijyra
├── git
│   ├── mammoth
│   ├── mammoth-helper
│   └── mammoth-hf
└── venv
    └── mammoth-hf
```

Now you can follow up to
[this page](https://github.com/Helsinki-NLP/mammoth/tree/feat/helper/helper/bin/slurm)
to learn how to create a job directory and use the installed software with it.
