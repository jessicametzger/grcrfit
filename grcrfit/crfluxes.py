import numpy as np

from . import physics as ph

# spl = single power law
# bpl = beta power law (p.l. w.r.t. momentum and beta)

# WITH MODULATION

def flux_spl(LIS_params, phi, E_TOA, Z, M):
    LIS_norm, alpha1 = LIS_params

    # figure out which interstellar energies/momenta you'll need
    E_IS=ph.demod_energy(E_TOA, phi, Z)
    p_IS=ph.E_to_p(E_IS, M)

    # construct true LIS at the same energies (momenta)
    flux_IS=LIS_norm*(p_IS**alpha1)

    # modulate the LIS according to phi
    flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, M)
    
    return flux_TOA_model
    

def flux_bpl(LIS_params, phi, E_TOA, Z, M):
    LIS_norm, alpha1, alpha2 = LIS_params
    
    # figure out which interstellar energies/momenta you'll need
    E_IS=ph.demod_energy(E_TOA, phi, Z)
    p_IS=ph.E_to_p(E_IS, M)
    E_tot=ph.get_Etot(E_IS, M)
    v_IS=p_IS*ph.C_SI/E_tot #units: c (so this is beta)

    # construct true LIS at the same energies (momenta)
    flux_IS=LIS_norm*(p_IS**alpha1)*(v_IS**alpha2)

    # modulate the LIS according to phi
    flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, M)
    
    return flux_TOA_model


# BELOW - INTERSTELLAR: NO MODULATION

def flux_spl_IS(LIS_params, p, M):
    LIS_norm, alpha1 = LIS_params
    p=np.array(p)

    # construct true LIS at the same energies (momenta)
    flux=LIS_norm*(p**alpha1)
    
    return flux
    
def flux_bpl_IS(LIS_params, p, M):
    LIS_norm, alpha1, alpha2 = LIS_params
    p=np.array(p)
    
    E_tot=ph.p_to_Etot(p,M)
    v=p*ph.C_SI/E_tot #units: c (so this is beta)

    # construct true LIS at the same energies (momenta)
    flux=LIS_norm*(p**alpha1)*(v**alpha2)
    
    return flux
    