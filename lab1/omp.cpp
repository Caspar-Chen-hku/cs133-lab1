// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...
/*
void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  // Your code goes here...
  #pragma omp parallel for
  for (int i = 0; i < kI; ++i) {
    std::memset(c[i], 0, sizeof(float) * kJ);
  }
  for (int i = 0; i < kI; ++i) {
    for (int j = 0; j < kJ; ++j) {
      #pragma omp parallel for reduction (+:c[i][j])
      for (int k = 0; k < kK; ++k) {
        c[i][j] += a[i][k] * b[k][j];
      }
    }
  }
}
*/

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
   int i,j,k;
#pragma omp parallel shared(a,b,c) private(i,j,k) 
   {
  #pragma omp for schedule(static)
    for (i=0; i< kI; i++){
      std::memset(c[i], 0, sizeof(float) * kJ);
        for (k=0; k< kK; k++){
          for (j=0; j< kJ; j++){
              c[i][j] += a[i][k] * b[k][j];
          }
        }
    }
   }
}