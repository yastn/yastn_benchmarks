# Benchmarks of YASTN

This repository contains benchmarks for [YASTN](https://github.com/yastn/yastn).

## 2-site DMRG

We follow a benchmark framework adopted by the authors of [TeNPy](https://github.com/tenpy/tenpy) and [ITensor](https://github.com/ITensor/ITensors.jl). It considers a spin-1 Heisenberg chain of 100 sites, performing a number of 2-site DMRG sweeps with increasing MPS bond dimensions.

See, [TeNPy benchmarks](https://github.com/tenpy/tenpy_benchmarks) and [TeNPy and ITensor Comparisons](
https://itensor.github.io/ITensorBenchmarks.jl/dev/tenpy_itensor/index.html).

After installing [YASTN](https://github.com/yastn/yastn) and cloning this repository, our benchmark script can be run as
```
python bench_dmrg.py
```
A few control options are available from the command line
```
python bench_dmrg.py --help
```

We run this benchmark focusing on U(1)-symmetric tensors employing a workstation with Intel i9-10920 CPU and Nvidia RTX-3090 GPU.

![alt text](https://github.com/yastn/benchmarks/blob/main/results_dmrg/bench.png?raw=true)

We first run the test on a single CPU core to compare the execution times of the three libraries, obtaining comparable performance for the corresponding simulation setups. Those employ package versions are: YASTN v1.2, TeNPy v1.0.3,  ITensors v0.6.16.

Next, we run YASTN using NumPy and PyTorch backends across one and multiple cores and utilizing GPU. PyTorch tensors show systematic overhead visible for small bond dimensions, which becomes marginal for larger bond dimensions. For CUDA (GPU) computation, we delegate the SVDs to the CPU due to the poor performance of SVD decomposition on the GPU. Still, in this example, the timings of large bond-dimension sweeps employing GPU and multiple-core CPUs are dominated by the SVD.

We thank TeNPy developer Johannes Hauschild for the discussions.

## CTMRG contractions

We collect core elements of CTMRG update for tensor sizes motivated by the applications described in [YASTN release article](https://arxiv.org/abs/2405.12196). See `.\bench_ctm.py` for details of the test framework, with symmetric tensor structures gathered in the folder `.\input_shapes\`.

Exemplary execution of full CTMRG update benchmark
```
python bench_ctm.py -model CtmBenchUpdate -params 'dims=(4, 2)' -fname 'Heisenberg_U1_d=2_D=4_chi=30'
```

### J1-J2 square lattice dataset

CTMRG on realistic states, i.e., optimized U(1)-symmetric iPEPS of the J1-J2 model from [j1j2_ipeps_states](https://github.com/jurajHasik/j1j2_ipeps_states) (included as a git submodule), is available through `CtmBenchUpdateJ1J2`. Here, `-fname` points to a state inside `./j1j2_ipeps_states/`, and `chi` defaults to the value of `chi_opt` in the file name
```
git submodule update --init
python bench_ctm.py -model CtmBenchUpdateJ1J2 -fname 'single-site_pg-C4v-A1_internal-U1/j20.25/state_1s_A1_U1B_j20.25_D5_chi_opt101'
```

### Kagome J1-JD dataset

A second dataset of realistic states, the dense iPEPS of the J1-JD model on the Kagome lattice from [j1jD_kagome_ipeps_states](https://github.com/jurajHasik/j1jD_kagome_ipeps_states) (also included as a git submodule), is available through `CtmBenchUpdateKagome`. These states coarse-grain three spin-1/2's of an up-pointing triangle into a single physical index of dimension 8 and carry no internal symmetry, so they exercise the dense code path. The states are translationally invariant (single-site), hence the unit cell defaults to `dims=(1, 1)`; `chi` again defaults to `chi_opt` in the file name
```
git submodule update --init
python bench_ctm.py -model CtmBenchUpdateKagome -fname 'IPEPS/J1.0_JD0.0/IPEPS_J1.0_JD0.0_D5_chi_opt80'
```
States with `D` = 2, ..., 10 are available in `./j1jD_kagome_ipeps_states/IPEPS/J1.0_JD0.0/`; note that `chi_opt` varies between them, so a sweep over `D` needs a wildcard
```
python bench_ctm.py -model CtmBenchUpdateKagome -fname 'IPEPS/J1.0_JD0.0/IPEPS_J1.0_JD0.0_D[3-9]_chi_opt*'
```

For more options, see
```
python bench_ctm.py --help
```