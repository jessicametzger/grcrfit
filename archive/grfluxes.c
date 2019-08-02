/* grfluxes.c
 *
 * Calculate the gamma ray emissivity contribution (counts/s) at given energy Eg (GeV)
 * for a given proton flux Jp (counts/(GeV s mbarn)) at given proton temperature Tp (GeV)
 * and given that proton energy's bin width (GeV)
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "/home/jmetzger/cparamlib/cparamlib/cparamlib.h"
#include <string.h>
#include <stdlib.h>

/* args: Eg, Jp, Tp, width */
int main(int argc, char *argv[])
{
    double ds, spec_cont, Eg, Jp, Tp, width;
    PARAMSET params;
    
    sscanf(argv[1],"%lf",&Eg);
    sscanf(argv[2],"%lf",&Jp);
    sscanf(argv[3],"%lf",&Tp);
    sscanf(argv[4],"%lf",&width);

    /* make sure the parameter struct is empty */
    memset(&params, 0, sizeof(PARAMSET));

    /* calculate parameters for this Tp */
    gamma_param(Tp, &params);

    /* calculate total inclusive cross section */
    ds = sigma_incl_tot(ID_GAMMA, Eg, Tp, &params);

    /* multiply for spectrum contribution */
    spec_cont = ds*Jp*width;

    printf("%lf",spec_cont);
    return 0;
}
