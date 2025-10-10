#!/bin/python3

import argparse
import glob
import os
from pathlib import Path
import shutil
import subprocess
import re
import itertools
import tempfile


def shell_command(command: str, input: str = None) -> str:
    print(f"\n$ {command}\n")
    subprocess.run(command, shell=True, check=True, text=True, input=input)


def setup_directory(dir: Path):
    dir.mkdir(parents=True, exist_ok=True)


def get_logs(command: str, working_dir: Path) -> Path:
    output_log = working_dir / "output.log"
    assembly_file = working_dir / "assembly.s"

    if output_log.exists():
        print(f"Logs already exist at {output_log}, skipping...")
        return output_log

    setup_directory(working_dir)
    os.environ["ROCROLLER_SAVE_ASSEMBLY"] = "1"
    os.environ["ROCROLLER_ASSEMBLY_FILE"] = str(assembly_file)
    shell_command(f"{command} | tee {output_log}")
    print(f"Logs saved to {output_log}, assembly saved to {assembly_file}")
    return output_log


def get_kernel_name(command: str, working_dir: Path) -> str:
    output_log = get_logs(command, working_dir)

    with open(output_log, "r") as f:
        regex = r"Generating: (.*)..."
        for line in f:
            match = re.search(regex, line)
            if match:
                kernel_name = match.group(1)
                return kernel_name
    raise ValueError("Kernel name not found in log")


def get_disassembly(command: str, working_dir: Path) -> Path:
    output_disasm = working_dir / "disasm.txt"

    if output_disasm.exists():
        print(f"Disassembly already exists at {output_disasm}, skipping...")
        return output_disasm

    kernel_name = get_kernel_name(command, working_dir)

    gdb_script = f"""
        set pagination off
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
    print(f"Disassembly saved to {output_disasm}")


def get_rocprofv3_trace(command: str, working_dir: Path) -> Path:
    output_dir = working_dir / "rocprof"

    if output_dir.exists():
        print(f"Rocprof trace already exists at {output_dir}, skipping...")
        return output_dir

    setup_directory(working_dir)

    # Install rocprof-trace-decoder if not exists
    if not Path("/opt/rocm-7.1.0/lib/librocprof-trace-decoder.so").exists():
        with tempfile.TemporaryDirectory() as tmpdir:
            shell_command(
                f"wget https://github.com/ROCm/rocprof-trace-decoder/releases/download/0.1.2/rocprof-trace-decoder-ubuntu-22.04-0.1.2-Linux.deb -O {tmpdir}/trace-decoder.deb"
            )
            shell_command(f"sudo dpkg -i {tmpdir}/trace-decoder.deb")

    rocprofv3 = f"""/opt/rocm/bin/rocprofv3 \
-d {output_dir} \
--att \
--att-target-cu=1 \
--att-shader-engine-mask=0xFFFFFFFF \
--att-perfcounter-ctrl=1 \
--att-perfcounters=SQ_LDS_BANK_CONFLICT,SQ_LDS_IDX_ACTIVE,SQ_INST_LEVEL_LDS,SQ_ACCUM_PREV_HIRES"""

    command = f"{rocprofv3} -- {command}"
    dispatch_index = 2

    while True:
        shell_command(f"{rocprofv3} -- {command}")

        csv_files = glob.glob(
            f"{output_dir}/stats_ui_output_agent_*_dispatch_{dispatch_index}.csv"
        )
        assert (
            len(csv_files) == 1
        ), f"Expected one CSV file, found {len(csv_files)} for {dispatch_index=}"

        with open(csv_files[0], "r") as f:
            output = f.read()
        print(output)
        if len(output) > 77:  # More than just the header
            break

        shutil.rmtree(output_dir)

    print(f"rocprofv3 trace saved to {output_dir}")


def run_gdb(command: str, working_dir: Path):
    """Launch interactive GDB session. Calls get_kernel_name if needed."""
    kernel_name = get_kernel_name(command, working_dir)

    gdb_script = f"""
        set pagination off
        set breakpoint pending on
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
    subprocess.call(full)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Kernel analysis for executables",
    )

    parser.add_argument(
        "-w", "--working-dir", type=Path, help="Working directory for analysis output"
    )

    parser.add_argument(
        "-d", "--disasm", action="store_true", help="Generate disassembly"
    )

    parser.add_argument(
        "-g", "--gdb", action="store_true", help="Launch interactive GDB session"
    )

    parser.add_argument(
        "-r", "--rocprof", action="store_true", help="Run rocprofv3 trace analysis"
    )

    args, extra = parser.parse_known_args()
    args.command = " ".join(extra).split("--")[-1].strip()

    print(f"Command: {args.command}")

    return args


def main():
    args = parse_args()

    if args.disasm:
        get_disassembly(args.command, args.working_dir)

    if args.rocprof:
        get_rocprofv3_trace(args.command, args.working_dir)

    if args.gdb:
        run_gdb(args.command, args.working_dir)


if __name__ == "__main__":
    main()
