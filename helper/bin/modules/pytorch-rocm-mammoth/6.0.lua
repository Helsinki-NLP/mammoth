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

local singName = 'lumi-pytorch-rocm-6.2.4-python-3.12-pytorch-v2.7.1.sif'
local pytorchVersion = '2.7.1'
local loadTxt = capture('cat ' .. parent .. '/load-pytorch-rocm-mammoth.txt')

help(string.format([[
ROCm-enabled PyTorch version %s for Python and MAMMOTH venv

]], pytorchVersion))

family("python_ml_env")

prepend_path('PATH', pathJoin(gparent .. '/wrappers'))

setenv('SING_IMAGE', pathJoin(ggparent .. '/images/' .. singName))


setenv('NCCL_SOCKET_IFNAME', 'hsn0,hsn1,hsn2,hsn3')  -- use only high speed network

setenv('MIOPEN_DISABLE_CACHE', '1')  -- disable cache
setenv('MIOPEN_USER_DB_PATH', '')    -- disable userdb
--setenv('MIOPEN_USER_DB_PATH', '/tmp/miopen-userdb-' .. os.getenv('USER'))
--setenv('MIOPEN_CUSTOM_CACHE_DIR', '/tmp/miopen-cache-' .. os.getenv('USER'))

--setenv('CXI_FORK_SAFE', '1')  -- these seem to be needed for multi node (via Samuel Antao)
setenv('CXI_FORK_SAFE_HP', '1')
setenv('FI_CXI_DISABLE_CQ_HUGETLB', '1')

setenv('NCCL_NET_GDR_LEVEL', 'PHB')
setenv('RCCL_NET_GDR_LEVEL', 'PHB')       -- GPU Direct RDMA when GPUs & NICs share common Host Bridge
setenv('NCCL_ENABLE_DMABUF_SUPPORT', '1')

setenv('SLURM_MPI_TYPE', 'pmi2')

setenv('FI_HMEM','rocr')
setenv('FI_LOG_LEVEL','warn')
setenv('FI_LOG_PROV','cxi')
setenv('FI_PROVIDER','cxi')               -- This is only for Cray.  Avoid ambiguity with OFI
setenv('HSA_ENABLE_DEBUG','0')

--setenv('HSA_FORCE_FINE_GRAIN_PCIE','1')   -- Sets fine grained memory on ONLY if you need it

setenv('RCCL_DEBUG','INFO')
setenv('RCCL_ENABLE_DMABUF_PLUGIN','0')   -- Since containers lack device bindings for DMABUF
setenv('RCCL_MSCCL_ENABLE','1')           -- Already set in the LUMI supported container; just to emphasize
setenv('RCCL_TRACE_PLUGIN','1')

-- setenv('PLUGIN_DIR','/opt/aws-ofi-rccl') -- Insider the container; Does not find!
-- setenv('SINGULARITYENV_LD_LIBRARY_PATH', '/opt/aws-ofi-rccl:/usr/local/lib:/opt/rocm/lib/:/usr/local/lib/python3.11/dist-packages/faiss:/opt/cray/libfabric/1.15.2.0/lib64')

setenv('PLUGIN_DIR', os.getenv('PROJHOME') .. '/lib') -- Insider the container
prepend_path('SINGULARITY_LD_LIBRARY_PATH', '/opt/aws-ofi-rccl:' .. os.getenv('PROJHOME') .. '/lib')

setenv('SING_FLAGS', '-B /opt/cray --bind /bin/ip:/bin/ip --bind /usr/lib64/libmnl.so.0:/usr/lib64/libmnl.so.0 \
  --bind /opt/cray/libfabric/1.15.2.0/bin/fi_info:/bin/fi_info ')
-- -B /opt/rocm/lib/librccl.so:/usr/local/lib/python3.10/dist-packages/torch/lib/librccl.so')

setenv('SINGULARITY_CONTAINLIBS', '/usr/lib64/libcxi.so.1,/usr/lib64/libjson-c.so.3,/opt/rocm/lib/librocm_smi64.so.6')

-- ROCm/PyTorch environment hook (so users can `eval $WITH_CONDA`)
setenv("WITH_CONDA", "source /opt/conda/etc/profile.d/conda.sh && conda activate pytorch")

if (mode() == "load") then
   LmodMessage(loadTxt)
end
