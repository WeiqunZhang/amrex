# Discrete Fourier Transform<span id="Chap:FFT"></span>

AMReX provides support for parallel discrete Fourier transform. The implementation utilizes cuFFT, rocFFT, oneMKL and FFTW, for CUDA, HIP, SYCL and CPU builds, respectively. It also provides FFT based Poisson solvers.

