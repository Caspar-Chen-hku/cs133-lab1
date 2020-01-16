// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int k,j,k0,j0;
  int BLOCK_SIZE = 256;

  #pragma omp parallel for private(k,j,k0,j0) schedule(static) num_threads(8)
    for (int i=0; i< kI; i++){
        for (k=0; k< kK; k+=BLOCK_SIZE){
          for (j=0; j< kJ; j+=BLOCK_SIZE){
              for (k0=k; k0<k+BLOCK_SIZE; k0++){
                for (j0=j; j0<j+BLOCK_SIZE; j0++){
                  c[i][j0] += a[i][k0] * b[k0][j0];
                }
              }
          }
        }
  }

}
