/* spectrum.c
 *
 * Calculate the gamma ray spectrum in #/(GeV s)
 * given a proton LIS in #/(GeV m^2 s sr)
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "/home/jmetzger/cparamlib/cparamlib/cparamlib.h"
#include <string.h>
#include <stdlib.h>

/* preset bins of Tp, GeV */
double Tp[43] = {512.0e3, 362.0e3, 256.0e3, 181.0e3, 128.0e3, 90.5e3, 64.0e3, 45.3e3,
                 32.0e3, 22.6e3, 16.0e3, 11.3e3, 8.0e3, 5.66e3, 4.0e3, 2.8e3, 2.0e3,
                 1.41e3, 1.0e3, 707.0, 500.0, 354.0, 250.0, 177.0, 125.0, 88.4, 62.5,
                 44.2, 31.3, 22.1, 15.6, 11.1, 7.81, 5.52, 3.91, 2.76, 1.95, 1.38,
                 0.98, 0.82, 0.69, 0.58, 0.488};

int main(int argc, char* argv[])
{
    double ds, E;
    int i, j;
    PARAMSET params;
    FILE *file;

    /* make sure the parameter struct is empty */
    memset(&params, 0, sizeof(PARAMSET));
    
    file = fopen("/home/jmetzger/grcrfit/gammads.dat","w");
    fprintf(file, "# horizontal: Egamma [MeV]. Vertical: Tp [MeV]. Contents: sigma_incl_tot contribution\n0");
    for (i = 0; i < 43; i++) {
        /* calculate parameters for this Tp */
        gamma_param(Tp[i], &params);
            
        if (i==0) {
            for (j=0; j<51; j++) {
                E = pow(10.0, j*0.08 - 2.0);
                fprintf(file, ", %lf", E*1000.);
            }
        }
        fprintf(file, "\n%lf", Tp[i]*1000.);

        /* calculate the inclusive cross section in each bin of Egamma */
        for (j = 0; j < 51; j++) {
            
            /* the gamma-ray energy is taken in 51 bins, from 10 MeV to 100 GeV */
            E = pow(10.0, j*0.08 - 2.0);
            
            /* and add them together and add to rest of the spectrum */
            ds = sigma_incl_tot(ID_GAMMA, E, Tp[i], &params);

            /* store in spectrum */
            fprintf(file, ", %lf", ds);
        }
    }
    fclose(file);

    return 0;
}
