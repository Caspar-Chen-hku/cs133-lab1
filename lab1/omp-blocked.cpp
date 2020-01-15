// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int k,j,i0,k0,j0;
  int BLOCK_SIZE = 256;
  #pragma omp parallel for private(k,j,k0,j0)
    for (int i=0; i< kI; i++){
        std::memset(c[i], 0, sizeof(float) * kJ);
        for (k=0; k< kK; k+=BLOCK_SIZE){
          for (j=0; j< kJ; j+=BLOCK_SIZE){
            for (k0=0; k0<BLOCK_SIZE; k0++){
              for (j0=0; j0<BLOCK_SIZE; j0++){
                c[i][j+j0] += a[i][k+k0] * b[k+k0][j+j0];
              }
            }
          }
        }
  }
}
