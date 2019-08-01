/* spectrum.c
 *
 * Calculate the gamma ray spectrum in #/(GeV s)
 * given a proton LIS in #/(GeV m^2 s sr)
 *
 */

#include <stdbool.h>
#include <stdio.h>
#include <math.h>
#include "/users/cparamlib/cparamlib/cparamlib.h"
#include <string.h>
#include <stdlib.h>

double c=299792458; /* m/s */
double pi=3.14159265;

/* protons only */
double E_to_p(double E) {
    double Ekin=E*1e3; /*GeV to MeV*/
    double Etot = Ekin + 939; /*total energy*/
    double radical=pow(Etot,2) - pow(939,2);
    if (radical<0) 
	printf("%lf\n", E);

    double p = pow(c,-1)*pow(radical,.5);
    return p;
}

/* preset bins of Tp, GeV */
double Tp[43] = {512.0e3, 362.0e3, 256.0e3, 181.0e3, 128.0e3, 90.5e3, 64.0e3, 45.3e3,
                 32.0e3, 22.6e3, 16.0e3, 11.3e3, 8.0e3, 5.66e3, 4.0e3, 2.8e3, 2.0e3,
                 1.41e3, 1.0e3, 707.0, 500.0, 354.0, 250.0, 177.0, 125.0, 88.4, 62.5,
                 44.2, 31.3, 22.1, 15.6, 11.1, 7.81, 5.52, 3.91, 2.76, 1.95, 1.38,
                 0.98, 0.82, 0.69, 0.58, 0.488};

/* the following double arrays are used to store the calculated spectra */
double *spectrum;
double *spectrum_nd;
double *spectrum_diff;
double *spectrum_delta;
double *spectrum_res;

int main(int argc, char* argv[])
{
    double E, Enext, dE;
    double Jp, pl_index, Pp, pl_norm;
    double ds, ds_nd, ds_diff, ds_delta, ds_res;
    double f, f_nd, f_diff, f_delta, f_res;
    double width;
    int i, j;
    int flag; /* 1 for Dermer fig 1, 2 for Dermer fig 2, other for ours */
    PARAMSET params;
    FILE *file;
    char line[100];

    printf("Enter run flag (%d for Dermer fig 1, %d for Dermer fig 2, other for yours)\n",1,2);
    fgets(line, sizeof(line), stdin);
    sscanf(line, "%d", &flag);

    /* allocate memory for the spectrum (180 bins) */
    spectrum = (double*)malloc(180 * sizeof(double));
    spectrum_nd = (double*)malloc(180 * sizeof(double));
    spectrum_diff = (double*)malloc(180 * sizeof(double));
    spectrum_delta = (double*)malloc(180 * sizeof(double));
    spectrum_res = (double*)malloc(180 * sizeof(double));

    /* make sure they are empty */
    memset(spectrum, 0, 180 * sizeof(double));
    memset(spectrum_nd, 0, 180 * sizeof(double));
    memset(spectrum_diff, 0, 180 * sizeof(double));
    memset(spectrum_delta, 0, 180 * sizeof(double));
    memset(spectrum_res, 0, 180 * sizeof(double));

    /* make sure the parameter struct is empty */
    memset(&params, 0, sizeof(PARAMSET));

    if (flag != 1 && flag != 2) {
        /* pl_norm is the normalization of the LIS times 1e20 */
        pl_norm=550085049.48;
        /* pl_index is the power-law index of the proton spectrum */
        pl_index = -2.858;
    } else if (flag == 1) {
        /* try old, fig 1 (eq 3) LIS */ 
        pl_index = -2.75;
    } else if (flag == 2) {
	/* try old, fig 2 (eq 4) LIS*/
	pl_index = -2.85;
    }

    for (i = 0; i < 43; i++) {
        /* calculate parameters for this Tp */
        gamma_param(Tp[i], &params);

        /* set the width of the bin, default is 1 but increased sampling
           near pion production threshold requires reduced bin widths */
	width = Tp[i]*.348*1e3;
        if (i < 38)
            width = 1*width;
        else if (i == 38)
            width = .75*width;
        else if (i > 38)
            width = .5*width;

        /* width = Tp[i]-Tp[i+1]; */

	if (flag != 2 && flag != 1) {
            /* calculate the proton spectrum (power law) at given Tp [GeV] */
            Pp = E_to_p(Tp[i]);
	    Jp = pow(Pp, pl_index);
            Jp = pl_norm*(1e-20)*Jp; /* add normalization */
            Jp = (1e-34)*Jp; /* change m^-2 to mbarn^-1 */
            Jp = 4*pi*Jp; /* cancel out the sr^-1 */
	} else if (flag == 1) {
	    /* try old LIS */
	    Jp = 4*pi*(1e-30)*2.2*pow(Tp[i] + .939, pl_index);
	} else if (flag == 2) {
	    Jp = 4*pi*(1e-30)*2.2*pow(Tp[i], pl_index/2)*pow(Tp[i] + 2*.939, pl_index/2);
	}

        /* calculate the inclusive cross section in each bin of Egamma */
        for (j = 0; j < 180; j++) {
            /* the gamma-ray energy is taken in 180 bins, from 1 MeV to 1000 TeV */
            E = pow(10.0, j*0.05 - 3.0);

            /* calculate individual contributions */
            ds_nd = sigma_incl_nd(ID_GAMMA, E, Tp[i], &params);
            ds_diff = sigma_incl_diff(ID_GAMMA, E, Tp[i], &params);
            ds_delta = sigma_incl_delta(ID_GAMMA, E, Tp[i], &params);
            ds_res = sigma_incl_res(ID_GAMMA, E, Tp[i], &params);
            
            /* and add them together and add to rest of the spectrum */
            ds = ds_nd + ds_diff + ds_delta + ds_res;

            /* store in spectrum */
            spectrum[j] += ds*Jp*width;
            spectrum_nd[j] += ds_nd*Jp*width;
            spectrum_diff[j] += ds_diff*Jp*width;
            spectrum_delta[j] += ds_delta*Jp*width;
            spectrum_res[j] += ds_res*Jp*width;
        }
    }
    
    /* save to a file */
    if (flag != 1 && flag != 2) {
	file = fopen("our_spectrum.csv", "w");
    } else if (flag == 1) {
	file = fopen("dermer_spectrum1.csv","w");
    } else if (flag == 2) {
	file = fopen("dermer_spectrum2.csv","w");
    }

    fprintf(file, "#spectrum due to power-law proton index %.2f\n", pl_index);
    fprintf(file, "# E in GeV, E*flux in count/s");
    for (i = 0; i < 180; i++) {
        E = pow(10.0, i*0.05 - 3.0);
        f = spectrum[i];
        f_nd = spectrum_nd[i];
        f_diff = spectrum_diff[i];
        f_delta = spectrum_delta[i];
        f_res = spectrum_res[i];
        fprintf(file, "%e %e %e %e %e %e\n", E, f, f_nd, f_diff, f_delta, f_res);
    }
    fclose(file);

    /* free allocated memory */
    free(spectrum);
    free(spectrum_nd);
    free(spectrum_diff);
    free(spectrum_delta);
    free(spectrum_res);

    return;
}
