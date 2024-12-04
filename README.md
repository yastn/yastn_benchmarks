# Benchmarks of YASTN

This repository contains a few benchmarks for [YASTN](https://github.com/yastn/yastn).

## 2-site DMRG

We follow a benchmark framework employing 2-site DMRG set up by the authors of [TeNPy](https://github.com/tenpy/tenpy) and [ITensor](https://github.com/ITensor/ITensors.jl). It considers a spin-1 Heisenberg chain of 100 sites, performing a number of DMRG sweeps with increasing MPS bond dimensions.

See, [TeNPy benchmarks](https://github.com/tenpy/tenpy_benchmarks) and [TeNPy and ITensor Comparisons](
https://itensor.github.io/ITensorBenchmarks.jl/dev/tenpy_itensor/index.html).

Following the installation of [YASTN](https://github.com/yastn/yastn), and cloning of this repository, our benchmark script can be run as
```
python bench_dmrg.py
```
with control options available from the command line, see
```
python bench_dmrg.py --help
```

We run this benchmark focusing on U(1)-symmetric tensors employing a workstation with Intel i9-10920 CPU and Nvidia RTX-3090 GPU.

![alt text](https://github.com/yastn/benchmarks/blob/main/results_dmrg/bench.png?raw=true)



We first run the test on a single CPU core to compare the execution times of the three libraries, obtaining comparable performance. A systematic shift for YASTN is attributed to a different contraction strategy of an effective two-site Hamiltonian acting on a trial vector in Krylov eigenvalue solver. YASTN, in the current version, uses a generic contraction scheme that is optimal for single contraction and allows large physical spaces (e.g., appearing in PEPS contraction). The other two libraries employ a scheme with different contraction order, allowing for precomputation of part of the diagram that later repeats while building the Krylov space.

Next, we run YASTN using NumPy and PyTorch backends across one and multiple cores and utilizing GPU. PyTorch tensors show systematic overhead visible for small bond dimensions and better use of parallelism in the limit of large bond dimensions. For CUDA (GPU) computation, we delegate the SVDs to the CPU due to the poor performance of SVD decomposition on the GPU.

We thank TeNPY developer Johannes Hauschild for the discussions.

## CTMRG contractions

We collect core elements of CTMRG update for tensor sizes motivated by real-life applications described in [YASTN release article](https://arxiv.org/abs/2405.12196).

See `.\bench_contractions.py` for details, with tensor sizes gathered in `.\input_shapes\` folder.