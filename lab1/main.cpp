#include <cstring>

#include <iostream>

#include "gemm.h"

using std::clog;
using std::endl;

// random comments

int main(int argc, char** argv) {
  // Allocate memory on heap to avoid stack overflow.
  static float a[kI][kK];
  static float b[kK][kJ];
  static float c_base[kI][kJ];
  static float c[kI][kJ];

  bool sequential = false;
  bool parallel = false;
  bool parallel_blocked = false;
  for (int i = 0; i < argc; ++i) {
    if (strcmp(argv[i], "sequential") == 0) {
      sequential = true;
    }
    if (strcmp(argv[i], "parallel") == 0) {
      parallel = true;
    }
    if (strcmp(argv[i], "parallel-blocked") == 0) {
      parallel_blocked = true;
    }
  }


  Init(a, b);

  GemmBaseline(a, b, c_base);

  if (sequential) {
    clog << "\nRun sequential GEMM\n";
    Benchmark(&GemmSequential, a, b, c);
    if (Diff(c_base, c) != 0) {
      clog << "Baseline failed\n";
      return 2;
    }
  }

  bool fail = false;
  if (parallel) {
    clog << "\nRun parallel GEMM with OpenMP\n";
    Benchmark(&GemmParallel, a, b, c);
    if (Diff(c_base, c) != 0) {
      fail = true;
    }
  }

  if (parallel_blocked) {
    clog << "\nRun blocked parallel GEMM with OpenMP\n";
    for (int block_size_i=kI/128; block_size_i<=kI; block_size_i*=2){
      for (int block_size_j=kJ/128; block_size_j<=kJ; block_size_j*=2){
        for (int block_size_k=kK/128; block_size_k<=kK; block_size_k*=2){
          Benchmark(&GemmParallelBlocked, a, b, c, block_size_i, block_size_j, block_size_k);
        }
      }
    }
    
    if (Diff(c_base, c) != 0) {
      fail = true;
    }
  }

  if (fail) {
    return 1;
  }
  return 0;
}
