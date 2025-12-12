-- dirname helper (no external libs)
local function dirname(p)
  return (p:match("(.+)/[^/]+$")) or "."
end

-- Where is this modulefile?
local mf      = myFileName()           -- e.g. /path/to/modulefiles/foo/1.2.lua
local mfDir   = dirname(mf)            -- …/modulefiles/foo
local parent  = dirname(mfDir)         -- …/modulefiles
local gparent = dirname(parent)        -- …/ (parent-parent)
local ggparent = dirname(gparent)        -- …/ (parent-parent-parent)

local singName = '/appl/local/containers/sif-images/lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif'
local pytorchVersion = '2.7.1'

local loadTxt = capture('cat ' .. mfDir .. '/load-pytorch-rocm-mammoth.txt')
-- I should not have big/many non-lua files in a module directory, but this is harmless

help(string.format([[
ROCm-enabled PyTorch version with  local module loader / python wrapper

]], pytorchVersion))

family("python_ml_env")

prepend_path('PATH', pathJoin(gparent .. '/wrappers'))

setenv('SING_IMAGE', singName)

setenv('SING_FLAGS', '-B /bin/ip:/bin/ip -B /usr/lib64/libmnl.so.0:/usr/lib64/libmnl.so.0 -B /opt/cray/libfabric/1.15.2.0/bin/fi_info:/bin/fi_info -B /usr/lib64/libcurl.so.4:/usr/lib/libcurl.so.4')
--  REMOVE_CRAY_DEPS=rm -rf /opt/cray /opt/cray-deps /usr/lib64/libcxi.so*
-- -B /opt/rocm/lib/librccl.so:/usr/local/lib/python3.10/dist-packages/torch/lib/librccl.so')

setenv('SINGULARITY_CONTAINLIBS', '/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/opt/rocm/lib/librocm_smi64.so.6')

-- setenv("WITH_CONDA", "source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch")
-- ROCm/PyTorch environment hook (so users can `eval $WITH_CONDA`)
-- No more in use:
--    $ which python
--    /pfs/lustrep1/projappl/project_462000964/members/aylijyra/git/mammoth/helper/wrappers/python
--    $ which sing-bash
--    /pfs/lustrep1/projappl/project_462000964/members/aylijyra/git/mammoth/helper/wrappers/sing-bash
--    $ sing-bash
--    echo $WITH_CONDA
--    source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch
--    Singularity> source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch
--    bash: /opt/conda/etc/profile.d/conda.sh: No such file or directory
--    Singularity> less /opt/miniconda3/envs/pytorch/bin/activate
--    /opt/miniconda3/envs/pytorch/bin/activate: No such file or directory
--    Singularity> which python
--    /opt/miniconda3/envs/pytorch/bin/python
--    Singularity> python
--    Python 3.12.11 | packaged by Anaconda, Inc. | (main, Jun  5 2025, 13:09:17) [GCC 11.2.0] on linux


-- ############ from pytorch module #############
setenv('NCCL_SOCKET_IFNAME', 'hsn')  -- use only high speed network

setenv('MIOPEN_DISABLE_CACHE', '1')  -- disable cache
setenv('MIOPEN_USER_DB_PATH', '')    -- disable userdb
--setenv('MIOPEN_USER_DB_PATH', '/tmp/miopen-userdb-' .. os.getenv('USER'))
--setenv('MIOPEN_CUSTOM_CACHE_DIR', '/tmp/miopen-cache-' .. os.getenv('USER'))

setenv('CXI_FORK_SAFE', '1')  -- these seem to be needed for multi node (via Samuel Antao)
-- AYJ: commented out in pytorch - why?

setenv('CXI_FORK_SAFE_HP', '1')
setenv('FI_CXI_DISABLE_CQ_HUGETLB', '1')

setenv('NCCL_NET_GDR_LEVEL', 'PHB')
setenv('NCCL_ENABLE_DMABUF_SUPPORT', '1')

setenv('SLURM_MPI_TYPE', 'pmi2')

-- ############ new choices / overrides #############
-- setenv('SLURM_MPI_TYPE', 'pmix_v4')
-- AYJ: On LUMI prefer PMIx via srun; module load cray-mpich for multi-rank DDP/torchrun
-- For PyTorch/MAMMOTH multi-node runs: allocate normally, then launch your job with srun
-- --mpi=pmix -n <tasks> ... so RCCL/OFI can use the PMI context. For single-rank testing, --mpi=none is fine.

setenv('NCCL_DEBUG','INFO')

setenv('FI_PROVIDER','cxi')                  -- This is only for Cray.  Avoid ambiguity with OFI
setenv('FI_HMEM','rocr')
setenv('FI_LOG_LEVEL','warn')
setenv('FI_LOG_PROV','cxi')
setenv('NCCL_SOCKET_IFNAME', 'hsn0,hsn1,hsn2,hsn3')  -- use only high speed network

-- setenv('RCCL_ENABLE_OFI','1')            -- Usually the OFI net plugin (aws-ofi-rccl) is
   -- auto-detected when on LD_LIBRARY_PATH.
   -- This toggle is non-standard; only use if your build supports it and autodetect fails.
   -- If unsure, prefer ensuring the plugin is visible instead of forcing this.
setenv('HSA_ENABLE_DEBUG','0')
--setenv('HSA_FORCE_FINE_GRAIN_PCIE','1')   -- Sets fine grained memory on ONLY if you need it
setenv('RCCL_DEBUG','INFO')
-- setenv('RCCL_ENABLE_DMABUF_PLUGIN','0')  -- Not supported
-- setenv('RCCL_MSCCL_ENABLE','1')          -- RCCL on LUMI doesn’t rely on MSCCL. 
setenv('RCCL_TRACE_PLUGIN','1')

-- ############ providing rocm, librccl-net-ofi etc. ################

setenv('SINGULARITYENV_LD_LIBRARY_PATH', '/opt/aws-ofi-rccl:/usr/local/lib:/opt/rocm/lib/:/usr/local/lib/python3.11/dist-packages/faiss:/opt/cray/libfabric/1.15.2.0/lib64')
prepend_path('SINGULARITYENV_LD_LIBRARY_PATH', gparent .. '/lib')  -- this contains alternative names referring to it

if (mode() == "load") then
   LmodMessage(loadTxt)
end
