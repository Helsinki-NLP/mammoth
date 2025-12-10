import os, sys, glob, ctypes, ctypes.util, sysconfig
# fast: does not import torch

# ---------- helpers ----------
def split_paths(envvar):
    return [p for p in os.environ.get(envvar, "").split(":") if p]

def unique(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out

def default_search_paths():
    paths = []
    # LD paths first
    paths += split_paths("LD_LIBRARY_PATH")
    # Conda/venv sites
    for key in ("CONDA_PREFIX","VIRTUAL_ENV","MESON_INSTALL_PREFIX"):
        p=os.environ.get(key)
        if p: paths.append(os.path.join(p,"lib"))
    # Python’s prefix libs
    for key in ("platlib","purelib"):
        try:
            paths.append(sysconfig.get_path(key))
        except KeyError:
            pass
    # ROCm common locations
    rocm = os.environ.get("ROCM_PATH","/opt/rocm")
    paths += [os.path.join(rocm,"lib"), os.path.join(rocm,"lib64")]
    paths += glob.glob("/opt/rocm-*/lib") + glob.glob("/opt/rocm-*/lib64")
    # System fallbacks
    paths += ["/usr/local/lib","/usr/local/lib64","/usr/lib","/usr/lib64","/lib","/lib64"]
    return [p for p in unique(paths) if p and os.path.isdir(p)]

def find_candidates(basename, search_paths):
    """Return list of existing files that match basename or basename.* in the paths."""
    hits=[]
    for d in search_paths:
        for pat in (basename, basename+".*", basename.replace(".so",".so.*")):
            hits += glob.glob(os.path.join(d, pat))
    return unique(hits)

def can_dlopen(libname):
    """Try dlopen by soname (using the dynamic loader’s rules)."""
    try:
        ctypes.CDLL(libname)
        return True, libname
    except OSError:
        # Try resolved path via ldconfig database
        found = ctypes.util.find_library(libname.replace("lib","").replace(".so",""))
        if found:
            try:
                ctypes.CDLL(found)
                return True, found
            except OSError:
                return False, found
        return False, None

def env_wants_ofi():
    # Heuristics: explicit RCCL_NET mentions of ofi, or any FI_* env suggests libfabric/OFI usage
    rccl_net = os.environ.get("RCCL_NET","")
    if "ofi" in rccl_net.lower(): return True
    # Some stacks mirror NCCL-style hints
    if "ofi" in os.environ.get("NCCL_NET","").lower(): return True
    if any(k.startswith("FI_") for k in os.environ.keys()): return True
    return False

def summarize(title, ok, loadable, paths):
    icon = "✔" if ok else "✖"
    print(f"{icon} {title}: {'found' if ok else 'NOT found'}; dlopen: {'OK' if loadable else 'FAIL'}")
    if paths:
        for p in paths[:5]:
            print(f"   - {p}")
        if len(paths) > 5:
            print(f"   … and {len(paths)-5} more")

# ---------- main ----------
def main():

    print("Running plugin-test.py...")

    want_ofi = env_wants_ofi()
    sought = "librccl-net-ofi.so" if want_ofi else "librccl-net.so"

    print("=================== RCCL net plugin check ======================")
    print(f"Python             : {sys.executable}")
    print(f"LD_LIBRARY_PATH    : {os.environ.get('LD_LIBRARY_PATH','')}")
    print(f"LIBFABRIC_PATH     : {os.environ.get('LIBFABRIC_PATH','')}")
    print(f"ROCM_PATH          : {os.environ.get('ROCM_PATH','')}")
    print(f"PYTORCH_ROCM_ARCH  : {os.environ.get('PYTORCH_ROCM_ARCH','')}")
    print(f"FI_* in ENV        : {any(k.startswith('FI_') for k in os.environ)}")
    print(f"Appears seeking    : {sought}")
    print("==================Searching for the Plugin======================")

    search_paths = default_search_paths()

    # Check both libraries
    results = {}
    for lib in ("librccl-net.so","librccl-net-ofi.so"):
        files = find_candidates(lib, search_paths)
        loadable, via = can_dlopen(lib)
        results[lib] = {"files": files, "loadable": loadable, "via": via}

    # Print summaries
    for lib in ("librccl-net.so","librccl-net-ofi.so"):
        info = results[lib]
        summarize(lib, bool(info["files"]), info["loadable"], info["files"])

    # Extra hint: OFI also requires libfabric
    if want_ofi:
        lf_found = any(find_candidates("libfabric.so", search_paths))
        print(f"✔ OFI requested → libfabric present: {'yes' if lf_found else 'no'}")
        if not lf_found:
            print("  Hint: install/provide libfabric (libfabric.so) in your library path.")

    # Exit code logic: fail if the sought one isn’t dlopen-able
    sought_ok = results[sought]["loadable"]
    if not sought_ok:
        print(f"❌ The sought library ({sought}) is NOT loadable with the current environment.")
        print("   Try adjusting LD_LIBRARY_PATH (or module load), or ensure the file exists in known lib dirs.")
        print("================================================================")
        return 2

    print(f"✅ The sought library ({sought}) is loadable. You’re good to go.")
    print("================================================================")
    return 0

if __name__ == "__main__":
    sys.exit(main())

