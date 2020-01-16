// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
    int k,j,total;
  #pragma omp parallel for private(k,j) schedule(static) num_threads(8)
    for (int i=0; i< kI; i++){
        std::memset(c[i], 0, sizeof(float) * kJ);
        for (j=0; j< kJ; j++){
          for (k=0; k< kK; k++){
              c[i][j] += a[i][k] * b[k][j];
          }
        }
    }
}