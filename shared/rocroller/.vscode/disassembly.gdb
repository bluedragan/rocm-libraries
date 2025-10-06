set $function_name = "GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel"
set $output_file = "disassembly.txt"

set breakpoint pending on
eval "break %s", $function_name
run

eval "set logging file %s", $output_file
set logging enabled on

eval "disassemble %s", $function_name

set logging enabled off
quit
