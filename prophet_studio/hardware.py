"""Detection du materiel : GPU (NVML, sinon nvidia-smi), CPU, RAM, systeme ; metriques temps reel.

La RTX 5060 (Blackwell, sm_120, 8 Go GDDR7, 448 Go/s) est la cible : elle impose un runtime CUDA >= 12.8
(pilote >= 570) et un budget VRAM serre ; tout le planificateur part de ces chiffres.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field

# nom (sous-chaine, du plus specifique au plus general) -> (architecture, compute capability, bande passante Go/s)
KNOWN_GPUS: list[tuple[str, str, str, int]] = [
    ("RTX 5090", "blackwell", "12.0", 1792), ("RTX 5080", "blackwell", "12.0", 960),
    ("RTX 5070 Ti", "blackwell", "12.0", 896), ("RTX 5070", "blackwell", "12.0", 672),
    ("RTX 5060 Ti", "blackwell", "12.0", 448), ("RTX 5060 Laptop", "blackwell", "12.0", 384),
    ("RTX 5060", "blackwell", "12.0", 448), ("RTX 5050", "blackwell", "12.0", 320),
    ("RTX 4090", "ada", "8.9", 1008), ("RTX 4080", "ada", "8.9", 717), ("RTX 4070 Ti", "ada", "8.9", 504),
    ("RTX 4070", "ada", "8.9", 504), ("RTX 4060 Ti", "ada", "8.9", 288), ("RTX 4060 Laptop", "ada", "8.9", 256),
    ("RTX 4060", "ada", "8.9", 272), ("RTX 4050", "ada", "8.9", 192),
    ("RTX 3090", "ampere", "8.6", 936), ("RTX 3080", "ampere", "8.6", 760), ("RTX 3070", "ampere", "8.6", 448),
    ("RTX 3060 Ti", "ampere", "8.6", 448), ("RTX 3060", "ampere", "8.6", 360), ("RTX 3050", "ampere", "8.6", 224),
    ("RTX 2080", "turing", "7.5", 448), ("RTX 2070", "turing", "7.5", 448), ("RTX 2060", "turing", "7.5", 336),
    ("GTX 1080 Ti", "pascal", "6.1", 484), ("GTX 1660", "turing", "7.5", 192),
    ("A100", "ampere", "8.0", 2039), ("L4", "ada", "8.9", 300), ("L40S", "ada", "8.9", 864), ("H100", "hopper", "9.0", 3350),
]


@dataclass
class GPU:
    index: int
    name: str
    vendor: str = "nvidia"
    vram_total_mib: int = 0
    vram_used_mib: int = 0
    vram_free_mib: int = 0
    driver: str = ""
    cuda_version: str = ""          # version CUDA maximale supportee par le pilote
    compute_cap: str = ""
    arch: str = ""
    bandwidth_gbs: int = 0
    display_active: bool | None = None

    @property
    def is_blackwell(self) -> bool:
        try:
            return float(self.compute_cap or 0) >= 10.0
        except ValueError:
            return self.arch == "blackwell"


@dataclass
class HardwareInfo:
    os: str
    os_release: str
    arch: str
    cpu: str
    cpu_cores: int
    cpu_threads: int
    ram_total_gib: float
    ram_available_gib: float
    avx2: bool | None = None
    avx512: bool | None = None
    gpus: list[GPU] = field(default_factory=list)
    nvml: bool = False
    warnings: list[str] = field(default_factory=list)

    @property
    def gpu(self) -> GPU | None:
        """GPU principal : le NVIDIA avec le plus de VRAM."""
        return max(self.gpus, key=lambda g: g.vram_total_mib) if self.gpus else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["primary_gpu"] = asdict(self.gpu) if self.gpu else None
        if self.gpu:
            d["primary_gpu"]["is_blackwell"] = self.gpu.is_blackwell
        return d


def enrich(g: GPU) -> GPU:
    for key, arch, cc, bw in KNOWN_GPUS:
        if key.lower() in g.name.lower():
            g.arch = g.arch or arch
            g.compute_cap = g.compute_cap or cc
            g.bandwidth_gbs = g.bandwidth_gbs or bw
            break
    if not g.arch and g.compute_cap:
        major = int(float(g.compute_cap))
        g.arch = {12: "blackwell", 10: "blackwell", 9: "hopper", 8: "ampere", 7: "turing", 6: "pascal"}.get(major, "")
    return g


# ---- NVIDIA ----------------------------------------------------------------------------------------------------
def _nvml_gpus() -> list[GPU] | None:
    try:
        import pynvml  # nvidia-ml-py : quelques Ko, appelle nvml.dll / libnvidia-ml.so
    except Exception:
        return None
    try:
        pynvml.nvmlInit()
    except Exception:
        return None
    try:
        driver = pynvml.nvmlSystemGetDriverVersion()
        driver = driver.decode() if isinstance(driver, bytes) else driver
        cuda = ""
        try:
            v = int(pynvml.nvmlSystemGetCudaDriverVersion())
            cuda = f"{v // 1000}.{(v % 1000) // 10}"
        except Exception:
            pass
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            cc = ""
            try:
                ma, mi = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{ma}.{mi}"
            except Exception:
                pass
            disp = None
            try:
                disp = bool(pynvml.nvmlDeviceGetDisplayActive(h))
            except Exception:
                pass
            out.append(enrich(GPU(i, name, "nvidia", mem.total // 2**20, mem.used // 2**20, mem.free // 2**20, driver, cuda, cc,
                                  display_active=disp)))
        return out
    except Exception:
        return None


SMI_FIELDS = "index,name,memory.total,memory.used,memory.free,driver_version,compute_cap,display_active"


def parse_nvidia_smi_csv(csv_text: str, cuda_version: str = "") -> list[GPU]:
    """Sortie de `nvidia-smi --query-gpu=<SMI_FIELDS> --format=csv,noheader,nounits`."""
    out = []
    for line in csv_text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6 or not parts[0].isdigit():
            continue
        def num(x):
            try:
                return int(float(x))
            except ValueError:
                return 0
        cc = parts[6] if len(parts) > 6 and re.match(r"^\d+\.\d+$", parts[6]) else ""
        disp = None
        if len(parts) > 7 and parts[7] in ("Enabled", "Disabled"):
            disp = parts[7] == "Enabled"
        out.append(enrich(GPU(int(parts[0]), parts[1], "nvidia", num(parts[2]), num(parts[3]), num(parts[4]), parts[5], cuda_version, cc,
                              display_active=disp)))
    return out


def parse_cuda_version(smi_header: str) -> str:
    m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", smi_header)
    return m.group(1) if m else ""


def _run(cmd: list[str], timeout: float = 8.0) -> str:
    kw = {}
    if sys.platform == "win32":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW : pas de console qui clignote
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw).stdout
    except Exception:
        return ""


def _smi_gpus() -> list[GPU]:
    smi = shutil.which("nvidia-smi")
    if not smi and sys.platform == "win32":
        cand = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe")
        smi = cand if os.path.exists(cand) else None
    if not smi:
        return []
    cuda = parse_cuda_version(_run([smi]))
    txt = _run([smi, f"--query-gpu={SMI_FIELDS}", "--format=csv,noheader,nounits"])
    if not txt.strip() or "not a valid field" in txt.lower():
        txt = _run([smi, "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader,nounits"])
    return parse_nvidia_smi_csv(txt, cuda)


# ---- CPU / RAM -------------------------------------------------------------------------------------------------
def _cpu_name() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8", errors="ignore"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "win32":
        try:
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except Exception:
            pass
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor()
    return platform.processor() or platform.machine()


def _cpu_flags() -> tuple[bool | None, bool | None]:
    if sys.platform.startswith("linux"):
        try:
            txt = open("/proc/cpuinfo", encoding="utf-8", errors="ignore").read()
            flags = set(re.findall(r"^flags\s*:\s*(.*)$", txt, re.M)[0].split()) if "flags" in txt else set()
            return "avx2" in flags, any(f.startswith("avx512") for f in flags)
        except (OSError, IndexError):
            return None, None
    return None, None


def _ram() -> tuple[float, float]:
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total / 2**30, vm.available / 2**30
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        info = {}
        for line in open("/proc/meminfo", encoding="utf-8"):
            k, v = line.split(":", 1)
            info[k] = int(v.split()[0]) / 2**20
        return info.get("MemTotal", 0.0), info.get("MemAvailable", 0.0)
    return 0.0, 0.0


def _cores() -> tuple[int, int]:
    threads = os.cpu_count() or 1
    try:
        import psutil
        return psutil.cpu_count(logical=False) or max(1, threads // 2), threads
    except Exception:
        return max(1, threads // 2), threads


def fake_gpus(spec: str) -> list[GPU]:
    """PROPHET_FAKE_GPU="NVIDIA GeForce RTX 5060:8151:650[:display]" : demonstrations et tests sans carte."""
    out = []
    for i, item in enumerate(x for x in spec.split(";") if x.strip()):
        parts = item.split(":")
        total, used = int(parts[1]), int(parts[2]) if len(parts) > 2 else 0
        out.append(enrich(GPU(i, parts[0], "nvidia", total, used, total - used, "580.88", "13.0",
                              display_active=(len(parts) > 3 and parts[3] == "display"))))
    return out


def detect() -> HardwareInfo:
    total, avail = _ram()
    cores, threads = _cores()
    avx2, avx512 = _cpu_flags()
    if os.environ.get("PROPHET_FAKE_GPU"):
        gpus, nvml = fake_gpus(os.environ["PROPHET_FAKE_GPU"]), False
    else:
        gpus = _nvml_gpus()
        nvml = gpus is not None
    if not gpus:
        gpus = _smi_gpus()
    info = HardwareInfo(os=platform.system(), os_release=platform.release(), arch=platform.machine().lower(), cpu=_cpu_name(),
                        cpu_cores=cores, cpu_threads=threads, ram_total_gib=round(total, 1), ram_available_gib=round(avail, 1),
                        avx2=avx2, avx512=avx512, gpus=gpus, nvml=nvml)
    if sys.platform == "darwin" and info.arch in ("arm64", "aarch64"):
        # Apple Silicon : memoire unifiee, Metal ; ~70 % de la RAM est allouable au GPU par defaut
        info.gpus = [GPU(0, "Apple Silicon (Metal)", "apple", int(total * 1024 * 0.7), 0, int(avail * 1024 * 0.7), arch="apple")]
    g = info.gpu
    if g and g.vendor == "nvidia":
        if g.is_blackwell and g.cuda_version:
            try:
                if float(g.cuda_version) < 12.8:
                    info.warnings.append(f"{g.name} (Blackwell) : pilote trop ancien (CUDA {g.cuda_version}) ; installez un pilote NVIDIA >= 570 (CUDA 12.8+).")
            except ValueError:
                pass
        if g.display_active:
            info.warnings.append(f"L'ecran est branche sur la {g.name} : Windows y reserve 0,3-1 Gio de VRAM. Brancher l'ecran sur la carte mere (iGPU) libere de la place pour le modele.")
    return info


# ---- metriques temps reel ------------------------------------------------------------------------------------------
class GpuMonitor:
    """Lecture legere (NVML) de l'utilisation, de la VRAM, de la temperature et de la puissance ; None sans GPU NVIDIA."""

    def __init__(self):
        self._h = None
        try:
            import pynvml
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                best, best_mem = None, -1
                for i in range(pynvml.nvmlDeviceGetCount()):
                    h = pynvml.nvmlDeviceGetHandleByIndex(i)
                    m = pynvml.nvmlDeviceGetMemoryInfo(h).total
                    if m > best_mem:
                        best, best_mem = h, m
                self._h, self._nv = best, pynvml
        except Exception:
            self._h = None

    def sample(self, own_pids: set[int] | None = None) -> dict | None:
        if self._h is None:
            return None
        nv, h = self._nv, self._h
        try:
            mem = nv.nvmlDeviceGetMemoryInfo(h)
            util = nv.nvmlDeviceGetUtilizationRates(h)
            out = {"util": util.gpu, "mem_util": util.memory, "vram_used_mib": mem.used // 2**20, "vram_total_mib": mem.total // 2**20}
            try:
                out["temp_c"] = nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU)
                out["power_w"] = round(nv.nvmlDeviceGetPowerUsage(h) / 1000.0, 1)
            except Exception:
                pass
            if own_pids is not None:
                ours = 0
                try:
                    for p in nv.nvmlDeviceGetComputeRunningProcesses(h):
                        if p.pid in own_pids and p.usedGpuMemory:
                            ours += p.usedGpuMemory // 2**20
                except Exception:
                    pass
                out["vram_ours_mib"] = ours
            return out
        except Exception:
            return None
