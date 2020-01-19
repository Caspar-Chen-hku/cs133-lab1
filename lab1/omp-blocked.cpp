// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int BLOCK_SIZE_I = kI/8;
  int BLOCK_SIZE_J = kJ/4;
  int BLOCK_SIZE_K = kK/64;

  #pragma omp parallel for schedule(static) num_threads(8)
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }

    #pragma omp parallel for schedule(static) num_threads(8)
    for (int i=0; i< kI; i+=BLOCK_SIZE_I){
        for (int k=0; k< kK; k+=BLOCK_SIZE_K){
          for (int j=0; j< kJ; j+=BLOCK_SIZE_J){
            int i0_limit = i+BLOCK_SIZE_I;
            for (int i0=i; i0<i0_limit; i0++){
              int k0_limit = k+BLOCK_SIZE_K;
              for (int k0=k; k0<k0_limit; k0++){
                int j0_limit = j+BLOCK_SIZE_J;
                for (int j0=j; j0<j0_limit; j0++){
                  c[i0][j0] += a[i0][k0] * b[k0][j0];
                }
              }
            }
          }
        }
  }

}
