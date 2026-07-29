#!/usr/bin/env bash
set -euo pipefail

# Single-node: launches bench_ctm.py. The unrolled-combo sum is split across all
# NUM_DEVICES * WORKERS_PER_DEVICE processes.
#
#   * single node:       ./run_patch_hubbard_dist.sh
#

# ---- benchmark configuration ----
# FNAME="Hubbard_U1xU1xZ2_d=4x4_D=12_chi=60"
FNAME="Heisenberg_U1_d=2_D=4_chi=30"
# ---- fermionic statistics ----
# Required for the Hubbard U1xU1xZ2 tensors (parity on the Z2 channel).
# MUST be blank for bosonic inputs or set to False for every symmetry.
# FERMIONIC="(False,False,True)"
FERMIONIC_FLAG=""
if [[ -n "${FERMIONIC:-}" ]]; then
  FERMIONIC_FLAG="-fermionic $FERMIONIC"
fi
MODEL="CtmBenchMeasureNconFermionic"
BACKEND="torch_cutensor"
TENSORDOT_POLICY="no_fusion"
# yastn config lazy_threshold; empty => yastn backend-default (0 for cuTensor, 0.5 otherwise).
# Override per-run, e.g. LAZY_THRESHOLD=0.3 ./run_patch_hubbard_mp.sh
LAZY_THRESHOLD="${LAZY_THRESHOLD:-}"
LAZY_THRESHOLD_FLAG=""
if [[ -n "$LAZY_THRESHOLD" ]]; then
  LAZY_THRESHOLD_FLAG="-lazy_threshold $LAZY_THRESHOLD"
fi
REPEAT=2                   # Number of benchmark repeats (for timing statistics)
DTYPE=float64
# NOTE: distributed=True only engages when 'unroll' is a dict — keep it in PARAMS.
#       See bench_ctm_dist.py for detailed description of the benchmark parameters.
# See https://yastn.github.io/yastn/tensor/largecontractions.html
PARAMS="dims=(2,2),sites=((0,0),(1,1)),checkerboard=True,\
checkpoint_loop=True,separate_layers=True,per_combo_path=False,\
optimizer_kwargs={'minimize':'write'},\
unroll={('v',1,0,'k'):1,('v',1,1,'k'):1}"

# ---- multi-gpu configuration (single-node) ----
DEVICES="['cuda:0',]" #'cuda:1']"
DEVICES_FLAG="-devices "$DEVICES
WORKERS_PER_DEVICE=1   # >0 routes to _oe_blocksparse_mp; 0 uses the multi-process path (or serial if $DEVICES is a single same-device entry)
NUM_DEVICES=$(echo "$DEVICES" | tr ',' '\n' | grep -cE 'cuda|cpu')

NUM_THREADS=1
export OMP_NUM_THREADS="$NUM_THREADS"
export MKL_NUM_THREADS="$NUM_THREADS"
export OPENBLAS_NUM_THREADS="$NUM_THREADS"


JOBNAME="${MODEL}_${BACKEND}_${TENSORDOT_POLICY}_${FNAME//=/}_mp${NUM_DEVICES}x${WORKERS_PER_DEVICE}"
RUNROOT=$(pwd)
JOBOUT="$RUNROOT/jobout"
mkdir -p "$JOBOUT"
LOGDIR="$JOBOUT/logs_${JOBNAME}/$(hostname -s)"
mkdir -p "$LOGDIR"

echo "Launching $JOBNAME on $(hostname -s): devices $DEVICES" \
     "(num_devices=$NUM_DEVICES x workers_per_device=$WORKERS_PER_DEVICE); logs under $LOGDIR"


# ---- environment settings ---
# export PYTHONPATH="<path-to-yastn>${PYTHONPATH:+:$PYTHONPATH}"     # if yastn is not installed 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export CUTENSOR_LOG_LEVEL=0
export YASTN_PROFILE=1                     # include nsys profiling info
export YASTN_META_CUTENSOR=CPU             # set to GPU for metadata precomputation at cost of memory fragmentation               
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # See https://pytorch.org/docs/stable/notes/cuda.html#cuda-memory-management
# export YASTN_OE_OOM_RETRY=0                            # and https://yastn.github.io/yastn/tensor/large_contractions.html for memory fragmentation mitigation
# export YASTN_OE_ALLOC_CONF=""               
# export YASTN_OE_CUDA_CACHE_RELEASE_LEVEL=0
export TAPP_LOG_LEVEL=0
# export TAPP_CACHE_STRICT=1               # Enforce cache key validation 
export CUTENSOR_BLOCKSPARSE_REPRODUCIBLE=0 # See cuTensor on binary reproducibility

PROFILE=()
if [[ "${NSYS:-0}" == "1" ]]; then
  PROFILE=(nsys profile
    -o "$LOGDIR/%h_p%p"              # %h = hostname, %p = process id → nsys variables. %h is unique per node on shared FS
    --force-overwrite=true
    -t cuda,nvtx,osrt   # NCCL shows up under cuda+nvtx
    --sample=none)                   # drop CPU sampling → smaller reports
fi

"${PROFILE[@]}" python3 -m pdb bench_ctm.py \
       -backend "$BACKEND" \
       -tensordot_policy "$TENSORDOT_POLICY" \
       -model "$MODEL" \
       -repeat "$REPEAT" \
       $FERMIONIC_FLAG \
       $LAZY_THRESHOLD_FLAG \
       $DEVICES_FLAG \
       -mp_workers_per_device $WORKERS_PER_DEVICE \
       -params "$PARAMS" \
       -fname "$FNAME" \
       -dtype "$DTYPE" -stdout
