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
        for (k=0; k< kK; k++){
          total = 0;
          //#pragma omp for reduction(+:total)
          for (j=0; j< kJ; j++){
              //c[i][j] += a[i][k] * b[k][j];
              total += a[i][k] * b[k][j];
          }
          //#pragma omp single
          c[i][j] += total;
        }
    }
}