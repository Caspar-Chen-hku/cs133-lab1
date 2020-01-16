// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int i,j,k0,j0;
  int BLOCK_SIZE = 256;
  /*
  #pragma omp parallel for private(k,j,i0,k0,j0) schedule(static) num_threads(8)
    for (int i=0; i< kI; i+=BLOCK_SIZE){
        std::memset(c[i], 0, sizeof(float) * kJ);
        for (k=0; k< kK; k+=BLOCK_SIZE){
          for (j=0; j< kJ; j+=BLOCK_SIZE){
            for (i0=i; i0<i+BLOCK_SIZE; i0++){
              for (k0=k; k0<k+BLOCK_SIZE; k0++){
                for (j0=j; j0<j+BLOCK_SIZE; j0++){
                  c[i0][j0] += a[i0][k0] * b[k0][j0];
                }
              }
            }
          }
        }
  }
  */
  #pragma omp parallel for private(j,i,k0,j0) schedule(static) num_threads(8)
  for (int k=0; k< kK; k+=BLOCK_SIZE){
      for (j=0; j< kJ; j+=BLOCK_SIZE){
        for (i=0; i< kI; i++){
          std::memset(c[i], 0, sizeof(float) * kJ);
          for (k0=k; k0<k+BLOCK_SIZE; k0++){
                for (j0=j; j0<j+BLOCK_SIZE; j0++){
                  c[i][j0] += a[i][k0] * b[k0][j0];
                }
          }
        }
    }
  }

}
