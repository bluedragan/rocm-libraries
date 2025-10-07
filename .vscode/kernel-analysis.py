#!/bin/python3

from pathlib import Path
import subprocess
import re
import itertools
import tempfile

COMMAND = """

client/rocroller-gemm --mac_m=64 --mac_n=64 --mac_k=64 --wave_m=-1 --wave_n=-1 --wave_k=-1 --wave_b=-1 --workgroup_size_x=-1 --workgroup_size_y=-1 --workgroupRemapXCC=False --workgroupRemapXCCValue=-1 --unroll_x=0 --unroll_y=0 --loadLDS_A=True --loadLDS_B=True --storeLDS_D=True --betaInFma=True --direct2LDS_A=False --direct2LDS_B=False --scheduler=Priority --schedulerCost=LinearWeighted --prefetch=True --prefetchInFlight=2 --prefetchLDSFactor=0 --prefetchMixMemOps=False --loadLDSScale_A=False --loadLDSScale_B=False --swizzleScale=False --prefetchScale=False --streamK=False --numWGs=0 --streamKTwoTile=False --streamKTwoTileDPFirst=False --matchMemoryAccess=True --M=3072 --N=4096 --K=4096 --alpha=2.0 --beta=0.5 --type_A=float --type_B=float --type_C=float --type_D=float --type_acc=float --trans_A=N --trans_B=N --scale_A=None --scaleType_A=None --scale_B=None --scaleType_B=None --scaleBlockSize=-1 --scaleSkipPermlane=False --scaleValue_A=1.0 --scaleValue_B=1.0 --workgroupMappingDim=-1 --workgroupMappingValue=-1 --num_warmup=1 --num_outer=1 --num_inner=10 --noCheck=False --visualize=False

""".strip()

WORKING_DIR = Path("./kernel-analysis")
OUTPUT_LOG = WORKING_DIR / "output.log"


def shell_command(command: str, input: str = None) -> str:
    print(f"Running command: {command}")
    return subprocess.check_output(command, shell=True, text=True, input=input)


def setup_directory(dir: Path):
    dir.mkdir(parents=True, exist_ok=True)


def get_logs(command: str, output_log: Path):
    shell_command(f"{command} | tee {output_log}")


def get_kernel_name(output_log: Path) -> str:
    with open(output_log, "r") as f:
        regex = r"Generating: (.*)..."
        for line in f:
            match = re.search(regex, line)
            if match:
                return match.group(1)
    raise ValueError("Kernel name not found in log")


def get_disassembly(command: str, kernel_name: str, output_disasm: Path):
    gdb_script = f"""
        set logging file {output_disasm}
        set breakpoint pending on
        break {kernel_name}
        run
        set logging enabled on
        disassemble {kernel_name}
        set logging enabled off
        quit
    """
    shell_command(f"rocgdb --args {command}", input=gdb_script)


def get_rocprofv3_trace(command: str, output_dir: Path):
    # Install rocprof-trace-decoder if not exists
    if not Path("/opt/rocm-7.1.0/lib/librocprof-trace-decoder.so").exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            shell_command(
                f"wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.2/rocprof-trace-decoder-ubuntu-22.04-0.1.2-Linux.deb -O {tmpdir}/trace-decoder.deb"
            )
            shell_command(f"sudo dpkg -i {tmpdir}/trace-decoder.deb")

    shell_command(
        f"/opt/rocm/bin/rocprofv3 --att -d {output_dir} --att-target-cu=1 --att-shader-engine-mask=0x1 -- {command}"
    )


def run_gdb(command: str, kernel_name: str):
    gdb_script = f"""
        set breakpoint pending on
        set pagination off
        break {kernel_name}
        run
        del 1
        set pagination on
    """.splitlines()
    ex_list = list(
        itertools.chain.from_iterable(
            [["-ex", line.strip()] for line in gdb_script if line.strip()]
        )
    )
    full = ["rocgdb", *ex_list, "--args", *command.split()]
    print(" ".join(full))
    subprocess.call(full)


if __name__ == "__main__":
    # setup_directory(WORKING_DIR)
    # get_logs(COMMAND, OUTPUT_LOG)
    kernel_name = get_kernel_name(OUTPUT_LOG)
    # get_disassembly(COMMAND, kernel_name, WORKING_DIR / "disasm.txt")
    get_rocprofv3_trace(COMMAND, WORKING_DIR / "rocprof")
    # run_gdb(COMMAND, kernel_name)
