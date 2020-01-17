// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallelBlocked(const float a[kI][kK], const float b[kK][kJ],
                         float c[kI][kJ]) {
  // Your code goes here...
  int BLOCK_SIZE = kI/16;

  #pragma omp parallel for private(k,j,i0,k0,j0) schedule(static) num_threads(6)
    for (int i=0; i< kI; i+=BLOCK_SIZE){
        for (int k=0; k< kK; k+=BLOCK_SIZE){
          for (int j=0; j< kJ; j+=BLOCK_SIZE){
            for (int i0=i; i0<i+BLOCK_SIZE; i0++){
              for (int k0=k; k0<k+BLOCK_SIZE; k0++){
                for (int j0=j; j0<j+BLOCK_SIZE; j0++){
                  c[i0][j0] += a[i0][k0] * b[k0][j0];
                }
              }
            }
          }
        }
  }

}
