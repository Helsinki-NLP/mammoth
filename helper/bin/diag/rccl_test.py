import os
import socket
import sys
import torch
import torch.distributed as dist
import subprocess
from pathlib import Path
import re

def is_inside_container():
    # Check common container markers
    try:
        with open("/proc/1/cgroup", "rt") as f:
            cgroup = f.read()
            if "docker" in cgroup:
                return "docker"
            if "apptainer" in cgroup:
                return "apptainer"
            if "singularity" in cgroup:
                return "singularity"
    except Exception:
        pass
    # Also check known Apptainer env vars
    if "APPTAINER_NAME" in os.environ:
        return "apptainer"
    if "SINGULARITY_NAME" in os.environ:
        return "singularity"
    return False

def check_binding_warning():
    print("\nrccl_test: 🔍 Container binding check:")
    container = is_inside_container()
    if container:
        print(f"rccl_test:     📦 Detected a _{container}_ container environment.")
        # Check if the bindings module was loaded (naively)
        binds_ok = (
            os.path.exists("/dev/kfd")
            and os.path.exists("/dev/dri")
            and any(os.path.isdir(p) for p in ["/opt/rocm", "/rocm"])
        )
        if not binds_ok:
            print("rccl_test:     ❌ It looks like `singularity-AI-bindings` was NOT loaded.")
            print("rccl_test:      You may be missing ROCm devices or drivers in this container.")
        else:
            print("rccl_test:     ✅ ROCm bindings (/dev/{kfd,dri} and /opt/rocm) appear to be available.")
    else:
        print("rccl_test:     ✅ Running outside container — assuming ROCm is still accessible.")
    try:
        result = subprocess.check_output(["hipcc", "--version"], stderr=subprocess.STDOUT, text=True)
        result = result.splitlines()[1].strip()
        print("rccl_test:     🔢 ROCm detected: " + result)
    except Exception:
        print("rccl_test:     ⚠️  Could not query ROCm HIP version (hipcc not found?)")
    return container


def check_loaded_plugin():
    print("\nrccl_test: 🔍 Verifying loaded RCCL plugin via environment:")

    plugin_dir = os.environ.get("PLUGIN_DIR", "(not set)")
    print(f"rccl_test:     PLUGIN_DIR = {plugin_dir}")

    libpath = os.environ.get("LD_LIBRARY_PATH", "(not set)")
    print(f"rccl_test:     LD_LIBRARY_PATH = {libpath}")

    plugin_name = "librccl-net-ofi.so"
    found = False

    for path in libpath.split(":"):
        candidate = os.path.join(path, plugin_name)
        if os.path.isfile(candidate):
            print(f"rccl_test:     ✅ Found RCCL OFI plugin at: {candidate}")
            found = True
            break

    if not found:
        print("rccl_test:     ❌ RCCL OFI plugin not found in LD_LIBRARY_PATH.")


import os
import subprocess
import re
import socket

import json, os, re, shutil, socket, subprocess

def _ip_bin():
    # Find ip(8) even if sbin isn't in PATH
    return shutil.which("ip") or "/sbin/ip" or "/usr/sbin/ip"

def get_all_host_ipv4s():
    """Collect all non-loopback IPv4s on the node without calling `hostname`."""
    ips = set()
    ip = _ip_bin()
    try:
        out = subprocess.check_output([ip, "-j", "-4", "addr"], text=True)
        data = json.loads(out)
        for link in data:
            for a in link.get("addr_info", []):
                if a.get("family") == "inet":
                    ips.add(a["local"])
    except Exception:
        # Fallback: use Python’s resolver (may return fewer addresses)
        try:
            _, _, iplist = socket.gethostbyname_ex(socket.gethostname())
            ips.update(iplist)
        except Exception:
            pass
    # Filter out loopback/link-local
    return {ip for ip in ips if not ip.startswith(("127.", "169.254."))}

       

def get_all_host_ipv4s_old():
    ips = set()
    # 1) All A records for our hostname
    try:
        _, _, iplist = socket.gethostbyname_ex(socket.gethostname())
        ips.update(iplist)
    except Exception:
        pass
    # 2) Anything the OS reports as assigned to this host
    try:
        out = subprocess.check_output("hostname -I", shell=True, text=True).split()
        ips.update(ip for ip in out if re.match(r"\d+\.\d+\.\d+\.\d+$", ip))
    except Exception:
        pass
    # Filter loopback/link-local if you want
    return {ip for ip in ips if not ip.startswith(("127.", "169.254."))}


def test_nccl_socket_ifname():
    print("rccl_test: 🔍 NCCL_SOCKET_IFNAME Diagnostic")

    prefix = os.environ.get("NCCL_SOCKET_IFNAME")
    if prefix is None:
        print("rccl_test:    ❌ NCCL_SOCKET_IFNAME is not set.")
        return
    print(f"rccl_test:    ✅ NCCL_SOCKET_IFNAME = {prefix}")

    # Get all interfaces
    try:
        iface_list = subprocess.check_output(
            "ip -o link show | awk -F': ' '{print $2}'",
            shell=True, text=True
        ).splitlines()
    except Exception as e:
        print(f"rccl_test:    ❌ Failed to list network interfaces: {e}")
        return

    #    matched = [iface for iface in iface_list if iface.startswith(prefix)]
    names = [p.strip() for p in os.environ["NCCL_SOCKET_IFNAME"].split(",") if p.strip()]
    matched = [iface for iface in iface_list
           if any(iface == n or iface.startswith(n) for n in names)]

    if not matched:
        print(f"rccl_test:    ❌ No interfaces found starting with '{prefix}'")
        return

    host_ips = get_all_host_ipv4s()
    hostname = socket.gethostname()
    print(f"rccl_test:      ℹ️  Hostname: {hostname}, node IPv4s: {', '.join(sorted(host_ips)) or 'none'}")
    
    for ifname in matched:
        print(f"\nrccl_test:    🔍 Checking interface: {ifname}")
        try:
            ip = _ip_bin()
            out = subprocess.check_output([ip, "-4", "-o", "addr", "show", "dev", ifname], text=True)
            m = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", out)
            if not m:
                print(f"rccl_test:      ⚠️  Interface '{ifname}' has no IPv4 address.")
                continue
            addr = m.group(1)
            print(f"rccl_test:      ✅ Interface '{ifname}' has IP: {addr}")

            if addr in host_ips:
                print("rccl_test:      ✅ Interface IP is one of the node's IPv4s — OK for NCCL.")
            else:
                print("rccl_test:      ℹ️  Interface IP differs from the hostname IPs — normal on multi-homed nodes.")

        except subprocess.CalledProcessError:
            print(f"rccl_test:    ❌ Failed to get IP info for interface '{ifname}'")

def check_dnabuf(container):
    print("\nrccl_test: 🔍 DMA Buffering Capacity Check:")
    results = {}
    conditions = 0
    
    # 2. ROCm version detection
    rocm_ver = 'Unknown'
    try:
        # Debian/Ubuntu style
        out = subprocess.check_output(
           "dpkg -l | grep -i rocm-core", 
           stderr=subprocess.DEVNULL, 
           universal_newlines=True, 
           shell=True)  # <-- fix: enable shell pipeline
        rocm_ver = out.split()[2]
    except subprocess.CalledProcessError:
        try:
            # RPM based systems
            out = subprocess.check_output(
               "rpm -qa | grep -i rocm-core",
               stderr=subprocess.DEVNULL,
               universal_newlines=True,
               shell=True)  # <-- also use shell here
            rocm_ver = out.strip()
        except subprocess.CalledProcessError:
            # Fallback: check rocm-smi tool
            try:
                out = subprocess.check_output(
                    "/opt/rocm/bin/rocm-smi --showdriverversion",
                    stderr=subprocess.DEVNULL, universal_newlines=True)
                rocm_ver = out.strip()
            except Exception:
                rocm_ver = 'Not found'
                # hipcc --version
                # one more way
    results['ROCm_version'] = rocm_ver
    if rocm_ver >= "6.0":
        print(f"rccl_test:     ✅ ROCm version {rocm_ver} > 6.0, as required by DMABUF")
        conditions = conditions + 1
    else:
        print(f"rccl_test:     ❌ ROCm version {rocm_ver} not > 6.0 but required by DMABUF")


    # 3. /dev/dma_heap check
    path = '/dev/dma_heap'
    if os.path.exists(path):
        mode = oct(os.stat(path).st_mode & 0o777)
        results['/dev/dma_heap'] = f'Exists, perms={mode}'
        conditions = conditions + 1
        print("rccl_test:     ✅ System may be DMA-BUF capable")
    else:
        results['/dev/dma_heap'] = 'Not present'
        print("rccl_test:     ❌ System is not DMA-BUF capable")

    if not container:
        conditions = conditions + 1
        print("rccl_test:     ✅ We are not inside a container")
    else:
        print("rccl_test:     ❌ We are in a container")

    # 1. NCCL_DMABUF_ENABLE
    val = os.environ.get("NCCL_DMABUF_ENABLE")
    if (val is None or val == "1") and (conditions == 3):
        results['NCCL_DMABUF_ENABLE'] = 'Not set'
        print("rccl_test:     ⚠️  NCCL_DMABUF_ENABLE=0 although all known conditions are met")
    if val == "1" and conditions < 3:
        results['NCCL_DMABUF_ENABLE'] = val
        print("rccl_test:     ⚠️  NCCL_DMABUF_ENABLE=1 but some conditions are not met")


    return results

import os
import torch
import subprocess

def check_fi_hmem():
    print("rccl_test: 🔍 FI_HMEM Diagnostic ===")

    # 1. Check environment variable
    fi_hmem = os.environ.get("FI_HMEM", "not set")
    print(f"rccl_test:    FI_HMEM = {fi_hmem}")

    if fi_hmem != "rocm":
        print("rccl_test:    ⚠️  FI_HMEM is not set to 'rocm' — memory registration for HMEM may be disabled.")
    else:
        print("rccl_test:    ✅ FI_HMEM=rocm is set — Libfabric HMEM support should be enabled.")

    # 2. Confirm torch + ROCm + GPU presence
    if not torch.cuda.is_available():
        print("rccl_test:    ❌ torch.cuda is not available — cannot run GPU collective test.")
        return

    try:
        dev = torch.device("cuda:0")
        x = torch.ones(10, device=dev)
        y = x * 2
        print(f"rccl_test:    ✅ Torch ROCm tensor ops working: y[0] = {y[0].item()}")
    except Exception as e:
        print(f"rccl_test:    ❌ Torch ROCm test failed: {e}")
        return

    # 3. Check `fi_info` for cxi & HMEM support
    try:
        out = subprocess.check_output("fi_info -p cxi -v", shell=True, text=True)
        if "FI_HMEM" in out:
            print("rccl_test:    ✅ fi_info confirms provider supports FI_HMEM")
        else:
            print("rccl_test:    ⚠️  fi_info output does not mention FI_HMEM — may not be supported")
    except Exception as e:
        print(f"rccl_test:    ❌ fi_info check failed: {e}")


def check_kernel_params():
    print("\nrccl_test: 🔍 Kernel parameter check:")
    try:
        with open("/proc/cmdline") as f:
            cmdline = f.read().strip().lower()

        hints = {
            "amd_iommu=on": ["amd_iommu=on"],
            "iommu=pt": ["iommu=pt", "iommu.passthrough=on"],
        }
        present = []
        missing = []
        for label, variants in hints.items():
            if not any(variant in cmdline for variant in variants):
                missing.append(label)
            else:
                present.append(label)
        if missing:
            print(f"rccl_test:     ⚠️  Kernel boot params not explicitly set: {', '.join(missing)}")
            print(f"rccl_test:     ✅ but these are probably implied by the present parameter: {', '.join(present)}")
        else:
            print("rccl_test:     ✅ Kernel boot params for RCCL look OK.")
    except Exception as e:
        print(f"rccl_test:     ⚠️  Failed to read /proc/cmdline: {e}")


def run_test():
    # os.system("env | egrep '^(RCCL|FI_|NCCL_|PLUGIN|LD_LIB|HSA_|WITH_CONDA|CXI_|SLURM_|MIOPEN_)'")
    os.environ.update({
        "MASTER_ADDR": "localhost",
        "MASTER_PORT": "12355",
        "RANK": "0",
        "WORLD_SIZE": "1",
        "RCCL_DEBUG": "INFO",
        "NCCL_DEBUG": "INFO",
        "FI_LOG_LEVEL": "INFO",
        "HSA_ENABLE_DEBUG": "1",
        "MIOPEN_ENABLE_LOGGING":"1",
        # "FI_PROVIDER": "cxi",
        # "FI_HMEM": "1",
        # "NCCL_NET_PLUGIN": "",
        # "RCCL_ENABLE_OFI": "1",
        # "FI_LOG_FILE": "test.fi.log",
        # "RCCL_LOG_FILE": "test.rccl.log"
    })

    
    print("\nrccl_test: 🚀 RCCL + OFI Communication Test (LUMI version)\n" + "rccl_test: " + "-"*60)
    if not torch.cuda.is_available():
        print("rccl_test:     ❌ No ROCm GPU detected by PyTorch.")
        sys.exit(1)
    print("rccl_test:     ✅ ROCm GPU successfully detected by PyTorch.")

    container = check_binding_warning()

    check_loaded_plugin()
    check_kernel_params()
    check_dnabuf(container)
    check_fi_hmem()
    test_nccl_socket_ifname()
    
    try:
        dist.init_process_group("nccl", init_method="env://")
        print("rccl_test: ✅ Successfully initialized RCCL backend.")
    except Exception as e:
        print(f"rccl_test: ❌ Failed to initialize distributed process group: {e}")
        return

    try:
        t = torch.tensor([42.0], device="cuda")
        dist.all_reduce(t)
        print(f"rccl_test:     🎯 all_reduce result: {t.item()} (42.0 is correct for one node)")
    except Exception as e:
        print(f"rccl_test:     ❌ all_reduce failed: {e}")

    dist.destroy_process_group()
    print("rccl_test: ✅ RCCL + OFI test completed.\n")

if __name__ == "__main__":
    run_test()
