## 2.4 Adjusting the MAMMOTH Requirements
In the following we show the development of the package requirements
1. the old requirements within the container and $WITH_CONDA only - packages missing
2. the old requirements with the virtual environ - the most similar packages
3. the same deemed under the updated requirements that will be added to `setup.py`

### Bare Bones Container Compatibility with the Old Requirements (mammoth_dep_check.py)
```bash
source init3.10-with-conda.sh
singularity exec $PROJHOME/sif/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif bash
$WITH_CONDA
python $PROJHOME/diag/mammoth_dep_check.py
deactivate; exit
```
| **Status** | **Package**         | **Version**        | **Location** | **Loaded** |
|------------|---------------------|--------------------|--------------|------------|
| ❌         | configargparse      | (not installed)   | - |
| ✅          | einops              | 0.8.0              (ok) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/einops/.` |
| ❌         | flake8              | (not installed)   | - |
| ❌         | flask               | (not installed)   | - |
| ❌         | pyonmttok           | (not installed)   | - |
| ❌         | pytest-flake8       | (not installed)   | - |
| ⚠️         | pytest              | 8.2.2              (not ==7.0.1) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/pytest/.` |
| ✅          | pyyaml              | 6.0.1              (ok) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/yaml/. | baseline` |
| ❌         | sacrebleu           | (not installed)   | - |
| ❌         | scikit-learn        | (not installed)   | - |
| ⚠️         | sentencepiece       | 0.2.0              (not ==0.1.97) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/sentencepiece/.` |
| ❌         | tensorboard         | (not installed)   | - |
| ❌         | timeout-decorator   | (not installed)   | - |
| ✅          | torch               | 2.3.0+rocm6.2.0    (ok) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/torch/.` |
| ⚠️         | tqdm                | 4.64.1             (not ==4.66.2) | `/opt/miniconda3/envs/pytorch/lib/python3.10/site-packages/tqdm/.` |
| ❌         | waitress            | (not installed)   | - |
| ❌         | x-transformers      | (not installed)   | - |

### Virtual Environment Compatibility with the Old Requirements (mammoth_dep_check.py)
```bash
source init3.10-with-conda.sh
singularity exec $PROJHOME/sif/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif bash
$WITH_CONDA
source $PROJHOME/venv/mammoth3.10-with-conda/bin/activate
python $PROJHOME/diag/mammoth_dep_check.py
deactivate; exit
```
| **Status** | **Package**         | **Version**        | **Location** | **Loaded** |
|------------|---------------------|--------------------|--------------|------------|
| ✅          | configargparse      | 1.7.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/configargparse.py` |
| ⚠️         | einops              | 0.8.1              (not ==0.8.0) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/einops/.` |
| ✅          | flake8              | 4.0.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/flake8/.` |
| ✅          | flask               | 2.0.3              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/flask/.` |
| ✅          | pyonmttok           | 1.37.1             (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pyonmttok/.` |
| ✅          | pytest-flake8       | 1.1.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pytest_flake8.py` |
| ✅          | pytest              | 7.0.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pytest/.` |
| ✅          | pyyaml              | 6.0.2              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/yaml/. | baseline` |
| ✅          | sacrebleu           | 2.3.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sacrebleu/.` |
| ⚠️         | scikit-learn        | 1.3.1              (not ==1.2.0) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sklearn/.` |
| ⚠️         | sentencepiece       | 0.2.0              (not ==0.1.97) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sentencepiece/.` |
| ✅          | tensorboard         | 2.19.0             (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/tensorboard/.` |
| ✅          | timeout-decorator   | 0.5.0              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/timeout_decorator/.` |
| ✅          | torch               | 2.7.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/torch/.` |
| ⚠️         | tqdm                | 4.67.0             (not ==4.66.2) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/tqdm/.` |
| ✅          | waitress            | 3.0.2              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/waitress/.` |
| ✅          | x-transformers      | 1.32.14            (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/x_transformers/.` |
### Virtual Environment Compatibility with the New Requirements (mammoth_dep_check_proposed3.11.py)
```bash
source init3.10-with-conda.sh
singularity exec $PROJHOME/sif/lumi-pytorch-rocm-6.2.0-python-3.10-pytorch-v2.3.0.sif bash
$WITH_CONDA
source $PROJHOME/venv/mammoth3.10-with-conda/bin/activate
python $PROJHOME/diag/mammoth_dep_check_proposed3.11.py
deactivate; exit
```
| **Status** | **Package**         | **Version**        | **Location** | **Loaded** |
|------------|---------------------|--------------------|--------------|------------|
| ✅          | configargparse      | 1.7.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/configargparse.py` |
| ✅          | einops              | 0.8.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/einops/.` |
| ✅          | flake8              | 4.0.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/flake8/.` |
| ✅          | flask               | 2.0.3              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/flask/.` |
| ✅          | pyonmttok           | 1.37.1             (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pyonmttok/.` |
| ✅          | pytest-flake8       | 1.1.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pytest_flake8.py` |
| ✅          | pytest              | 7.0.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/pytest/.` |
| ✅          | pyyaml              | 6.0.2              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/yaml/.` | baseline|
| ✅          | sacrebleu           | 2.3.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sacrebleu/.` |
| ✅          | scikit-learn        | 1.3.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sklearn/.` |
| ✅          | sentencepiece       | 0.2.0              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/sentencepiece/.` |
| ✅          | tensorboard         | 2.19.0             (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/tensorboard/.` |
| ✅          | timeout-decorator   | 0.5.0              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/timeout_decorator/.` |
| ✅          | torch               | 2.7.1              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/torch/.` |
| ✅          | tqdm                | 4.67.0             (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/tqdm/.` |
| ✅          | waitress            | 3.0.2              (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/waitress/.` |
| ✅          | x-transformers      | 1.32.14            (ok) | `$PROJHOME/venv/mammoth3.10-with-conda/lib/python3.10/site-packages/x_transformers/.` |

The last column of these printouts is for showing the layer where a particular package has been added.  This requires the use of some diagnostic commands such as `tracked_pip_install` and `tracked_module_load` defined separately by a script that defines bash functions.  Now, this feature is not in use. 