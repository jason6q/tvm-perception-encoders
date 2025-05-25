"""
    TVM Compilation pipeline
    See tutorial on how to format the pipeline: https://tvm.apache.org/docs/how_to/tutorials/optimize_llm.html
"""
from logging import getLogger
from typing import List

import tvm
import tvm.relax.frontend.nn as nn
import tvm.relax as R

logger = getLogger()

@R.register_pipeline("opt_pe")
def _pipeline( ext_mods: List[nn.ExternModule] = None, opt_level: int = 0):
    """
        Generate the compilation pipeline here. Involves 4 stages
        1. High-level operator graph conversions
        2. Lowering into TIR
        3. Passes over TIR to optimize further.
        4. Lowering to Bytecode for runtime.
    """

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        seq = tvm.transform.Sequential(
            [
            # Phase 1. High-level operator graph
            R.transform.FuseTransposeMatmul(),

            # Phase 2. Lower to TIR, this part is just TVM Relax's official zero pipeline.
            R.transform.LegalizeOps(),

            # Phase 3. Passes on TIR
            R.transform.DeadCodeElimination(),

            # Phase 4. Low-level Optimizations like dlight scheduling

            # Phase 5. Lowering to VM Bytecode.
            R.transform.RewriteDataflowReshape(),
            R.transform.ToNonDataflow(),
            R.transform.RemovePurityChecking(),
            R.transform.CallTIRRewrite(),
            R.transform.StaticPlanBlockMemory(),
            R.transform.RewriteCUDAGraph(),
            R.transform.LowerAllocTensor(),
            R.transform.KillAfterLastUse(),
            R.transform.LowerRuntimeBuiltin(),
            R.transform.VMShapeLower(),
            R.transform.AttachGlobalSymbol()]
        )

        mod = seq(mod)
        return mod
    return _pipeline


def compile(mod, device, opt_level: int = 3):
    """
        Build the executible with the compilation pipeline and return a virtual machine
        to run it.
    """
    logger.info(f"Compiling...")
    target = tvm.target.Target.from_device(device)
    with target:
        ex = tvm.compile(mod, target, relax_pipeline=R.get_pipeline("opt_pe"))
        vm = R.VirtualMachine(ex, device)
    return vm