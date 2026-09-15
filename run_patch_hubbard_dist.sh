#!/usr/bin/env bash
set -euo pipefail

# Single- AND multi-node: launches bench_ctm_dist.py with NPROC_PER_NODE ranks
# per node (one per GPU). The unrolled-combo sum is split across all
# world = NNODES * NPROC_PER_NODE ranks (LPT) and all_reduced across nodes via
# NCCL (see yastn._oe_blocksparse_dist).
#
# This is a PER-NODE launcher: it runs `torchrun` for the local node and relies
# on c10d (elastic) rendezvous to coordinate nodes — no explicit --node-rank
# needed, nodes discover each other via RDZV_ENDPOINT + RDZV_ID. Launch it:
#
#   * single node:       ./run_patch_hubbard_dist.sh
#   * multi-node, SLURM: srun --nodes=$NNODES --ntasks-per-node=1 ./run_patch_hubbard_dist.sh
#                        (srun starts this script once per node; SLURM env below
#                         auto-derives the head node and a shared RDZV_ID)
#   * multi-node, manual: run it on each node with the SAME NNODES, HEAD_NODE and
#                         RDZV_ID exported, e.g.
#                           NNODES=2 HEAD_NODE=node0 RDZV_ID=job123 ./run_patch_hubbard_dist.sh
#
# CPU/gloo sanity (no GPUs), 1 node, 2 ranks:
#   NNODES=1 NPROC_PER_NODE=2 DEVICE=cpu ./run_patch_hubbard_dist.sh

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
BACKEND="torch"
TENSORDOT_POLICY="no_fusion"
# yastn config lazy_threshold; empty => yastn backend-default (0 for cuTensor, 0.5 otherwise).
# Override per-run, e.g. LAZY_THRESHOLD=0.3 ./run_patch_hubbard_dist.sh
LAZY_THRESHOLD="${LAZY_THRESHOLD:-}"
LAZY_THRESHOLD_FLAG=""
if [[ -n "$LAZY_THRESHOLD" ]]; then
  LAZY_THRESHOLD_FLAG="-lazy_threshold $LAZY_THRESHOLD"
fi
DEVICE="${DEVICE:-cuda}"   # base device; each rank uses cuda:LOCAL_RANK (nccl) or cpu (gloo)
REPEAT=2                   # Number of benchmark repeats (for timing statistics)
DTYPE=float64
# NOTE: distributed=True only engages when 'unroll' is a dict — keep it in PARAMS.
#       See bench_ctm_dist.py for detailed description of the benchmark parameters.
# See https://yastn.github.io/yastn/tensor/largecontractions.html
PARAMS="dims=(2,2),sites=((0,0),(1,1)),checkerboard=True,\
checkpoint_loop=True,separate_layers=True,per_combo_path=False,\
optimizer_kwargs={'minimize':'write'},\
unroll={('v',1,0,'k'):1,('v',1,1,'k'):1}"

# ---- distributed configuration ----
# Ranks (GPUs) per node. Precedence:
#   1. explicit NPROC_PER_NODE export
#   2. SLURM GPU/task allocation (when running under SLURM)
#   3. local GPU count via nvidia-smi
#   4. 1
if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    NPROC_PER_NODE=$SLURM_GPUS_ON_NODE
  elif [[ -n "${SLURM_GPUS_PER_NODE:-}" ]]; then
    # may be "type:count" or "count"; take the trailing integer
    NPROC_PER_NODE=${SLURM_GPUS_PER_NODE##*:}
  elif [[ -n "${SLURM_NTASKS_PER_NODE:-}" ]]; then
    NPROC_PER_NODE=$SLURM_NTASKS_PER_NODE
  else
    NPROC_PER_NODE=$(nvidia-smi -L 2>/dev/null | wc -l)
  fi
fi
if [[ -z "$NPROC_PER_NODE" || "$NPROC_PER_NODE" -lt 1 ]]; then
  NPROC_PER_NODE=1
fi
# Number of nodes (SLURM_NNODES when under SLURM).
NNODES=${NNODES:-${SLURM_NNODES:-1}}
WORLD=$(( NNODES * NPROC_PER_NODE ))

# ---- c10d rendezvous ----
# All ranks rendezvous against HEAD_NODE:MASTER_PORT with a shared RDZV_ID.
# Under SLURM derive the head node from the allocation; otherwise default to
# localhost (single node) or an explicitly exported HEAD_NODE.
export MASTER_PORT="${MASTER_PORT:-29555}"
if [[ -z "${HEAD_NODE:-}" ]]; then
  if [[ -n "${SLURM_JOB_NODELIST:-}" ]] && command -v scontrol &>/dev/null; then
    HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
  else
    HEAD_NODE=127.0.0.1
  fi
fi
RDZV_ENDPOINT="${RDZV_ENDPOINT:-${HEAD_NODE}:${MASTER_PORT}}"
RDZV_ID="${RDZV_ID:-${SLURM_JOB_ID:-bench_$(date +%s)}}"

SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-${NUM_THREADS:-1}}
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"


# Per-rank log files: torchrun `-r 3` redirects each worker's stdout+stderr into
# $LOGDIR/<rdzv_id>/attempt_0/<local_rank>/{stdout,stderr}.log. Combined with
# `-stdout` on bench_ctm_dist.py, the benchmark output AND yastn's diagnostic
# logging (the dist per-rank combo/LPT-load line, routed to stdout via
# -log_level) land together in each rank's stdout.log.
#
# The hostname suffix keeps per-node local_rank subdirs from colliding when
# $JOBOUT is on a shared filesystem (each node has a local_rank 0).

JOBNAME="${MODEL}_${BACKEND}_${TENSORDOT_POLICY}_${FNAME//=/}_dist${NNODES}x${NPROC_PER_NODE}"
RUNROOT=$(pwd)
JOBOUT="$RUNROOT/jobout"
mkdir -p "$JOBOUT"
LOGDIR="$JOBOUT/logs_${JOBNAME}/$(hostname -s)"
mkdir -p "$LOGDIR"

echo "Launching $JOBNAME on $(hostname -s): nnodes=$NNODES x nproc_per_node=$NPROC_PER_NODE" \
     "(world=$WORLD); rdzv=$RDZV_ENDPOINT id=$RDZV_ID; per-rank logs under $LOGDIR"


# ---- environment settings ---
# export PYTHONPATH="<path-to-yastn>${PYTHONPATH:+:$PYTHONPATH}"     # if yastn is not installed 
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUTENSOR_LOG_LEVEL=0
export YASTN_PROFILE=1                     # include nsys profiling info
export YASTN_META_CUTENSOR_V=2              
# export YASTN_OE_OOM_RETRY=0                 
# export YASTN_OE_ALLOC_CONF=""               
# export YASTN_OE_CUDA_CACHE_RELEASE_LEVEL=0
export TAPP_LOG_LEVEL=0
# export TAPP_CACHE_STRICT=1               # Enforce cache key validation 
export CUTENSOR_BLOCKSPARSE_REPRODUCIBLE=0 # See cuTensor on binary reproducibility
# export NCCL_DEBUG=INFO                   # uncomment to debug inter-node NCCL connectivity

PROFILE=()
if [[ "${NSYS:-0}" == "1" ]]; then
  PROFILE=(nsys profile
    -o "$LOGDIR/%h_p%p"              # %h = hostname, %p = process id → nsys variables. %h is unique per node on shared FS
    --force-overwrite=true
    -t cuda,nvtx,osrt   # NCCL shows up under cuda+nvtx
    --sample=none)                   # drop CPU sampling → smaller reports
fi

"${PROFILE[@]}" torchrun \
       --nnodes="$NNODES" \
       --nproc-per-node="$NPROC_PER_NODE" \
       --rdzv-backend=c10d \
       --rdzv-endpoint="$RDZV_ENDPOINT" \
       --rdzv-id="$RDZV_ID" \
       -r 3 --log-dir "$LOGDIR" \
       bench_ctm_dist.py \
       -backend "$BACKEND" \
       -tensordot_policy "$TENSORDOT_POLICY" \
       -device "$DEVICE" \
       -model "$MODEL" \
       -repeat "$REPEAT" \
       $FERMIONIC_FLAG \
       $LAZY_THRESHOLD_FLAG \
       -params "$PARAMS" \
       -fname "$FNAME" \
       -dtype "$DTYPE" -log_level INFO -stdout
