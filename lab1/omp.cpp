// Header inclusions, if any...
#include <omp.h>
#include <cstring>

#include "gemm.h"

// Using declarations, if any...

/*
void GemmTrans(const float b[kK][kJ], float bt[kJ][kK]){
  int j;
  #pragma omp parallel for private(j) schedule(static) num_threads(8)
  for (j = 0; j < kK; j++)
	{
		for (int i = 0; i < kJ; i++)
		{
			bt[i][j] = b[j][i];
		}
	}
}
*/

void GemmParallel(const float a[kI][kK], const float b[kK][kJ],
                  float c[kI][kJ]) {
  #pragma omp parallel for //schedule(static) num_threads(8)
    for (int i=0; i< kI; i++){
      std::memset(c[i], 0, sizeof(float) * kJ);
        for (int j=0; j< kJ; j++){
          for (int k=0; k< kK; k++)
          {
              c[i][j] += a[i][k] * b[k][j];
          }
        }
    }
}