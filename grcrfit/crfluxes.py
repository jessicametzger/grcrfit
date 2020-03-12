# functions that generate CR fluxes given LIS parameters, energy values, particle masses/charges,
#  and sometimes solar modulation
# flux units match data units: #/sr/s/m2/(GeV/n)

import numpy as np

from . import physics as ph

np.seterr(all='raise')

# spl = single power law
# bpl = beta power law (p.l. w.r.t. momentum and beta)
# brpl = broken power law (as in Strong 2015, ICRC)
# dbrpl = double-broken power-law

# WITH MODULATION

def flux_spl(LIS_params, phi, E_TOA, el):
    LIS_norm, alpha1 = LIS_params

    try:
        # figure out which interstellar energies/momenta you'll need
        E_IS=ph.demod_energy(E_TOA, phi, ph.Z_DICT[el])
        p_IS=ph.E_to_p(E_IS, ph.M_DICT[el])
        if not (np.all(np.isfinite(E_IS)) and np.all(np.isfinite(p_IS))):
            return -np.inf

        # construct true LIS at the same energies (momenta)
        flux_IS = flux_spl_IS(LIS_params, p_IS, el)

        # modulate the LIS according to phi
        flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, ph.M_DICT[el])
    except:
        return -np.inf
    
    if np.any(flux_TOA_model<=0): return -np.inf
    
    return flux_TOA_model
    

def flux_bpl(LIS_params, phi, E_TOA, el):
    LIS_norm, alpha1, alpha = LIS_params
    
    try:
        # figure out which interstellar energies/momenta you'll need
        E_IS=ph.demod_energy(E_TOA, phi, ph.Z_DICT[el])
        p_IS=ph.E_to_p(E_IS, ph.M_DICT[el])
        if not (np.all(np.isfinite(E_IS)) and np.all(np.isfinite(p_IS))):
            return -np.inf
        E_tot=ph.get_Etot(E_IS, ph.M_DICT[el])
        v_IS=p_IS*ph.C_SI/E_tot #units: c (so this is beta)

        # construct true LIS at the same energies (momenta)
        flux_IS = flux_bpl_IS(LIS_params, p_IS, el)

        # modulate the LIS according to phi
        flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, ph.M_DICT[el])
    except:
        return -np.inf
    
    if np.any(flux_TOA_model<=0): return -np.inf
    
    return flux_TOA_model

def flux_brpl(LIS_params, phi, E_TOA, el):
    LIS_norm, alpha1, alpha2, pc_br, delta = LIS_params

    try:
        # figure out which interstellar energies/momenta you'll need
        E_IS=ph.demod_energy(E_TOA, phi, ph.Z_DICT[el])
        p_IS=ph.E_to_p(E_IS, ph.M_DICT[el])
        if not (np.all(np.isfinite(E_IS)) and np.all(np.isfinite(p_IS))):
            return -np.inf

        # construct true LIS at the same energies (momenta)
        flux_IS = flux_brpl_IS(LIS_params, p_IS, el)

        # modulate the LIS according to phi
        flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, ph.M_DICT[el])
    except:
        return -np.inf
    
    if np.any(flux_TOA_model<=0): return -np.inf
    
    return flux_TOA_model

def flux_dbrpl(LIS_params, phi, E_TOA, el):
    LIS_norm, alpha1, alpha3, pc_br2, delta2, alpha2, pc_br1, delta1 = LIS_params
    # independent among elements: LIS_norm, alpha1(HE), alpha3(LE), pc_br2(M-L), delta2(M-L)
    # common among elements: alpha2(ME), pc_br1(H-M), delta1(H-M)

    try:
        # figure out which interstellar energies/momenta you'll need
        E_IS=ph.demod_energy(E_TOA, phi, ph.Z_DICT[el])
        p_IS=ph.E_to_p(E_IS, ph.M_DICT[el])
        if not (np.all(np.isfinite(E_IS)) and np.all(np.isfinite(p_IS))):
            return -np.inf

        # construct true LIS at the same energies (momenta)
        flux_IS = flux_dbrpl_IS(LIS_params, p_IS, el)

        # modulate the LIS according to phi
        flux_TOA_model=ph.mod_flux(flux_IS, E_TOA, E_IS, ph.M_DICT[el])
    except:
        return -np.inf
    
    if np.any(flux_TOA_model<=0): return -np.inf
    
    return flux_TOA_model


# BELOW - INTERSTELLAR: NO MODULATION

def flux_spl_IS(LIS_params, p, el):
    LIS_norm, alpha1 = LIS_params
    p=np.array(p)

    try:
        # construct true LIS at the same energies (momenta)
        p_ref = ph.p10_DICT[el]
        flux=LIS_norm*((p/p_ref)**(-alpha1))
    except:
        return -np.inf
    
    if np.any(flux<=0): return -np.inf
    
    return flux
    
def flux_bpl_IS(LIS_params, p, el):
    LIS_norm, alpha1, alpha = LIS_params
    p=np.array(p)
    
    try:
        E_tot=ph.p_to_Etot(p,ph.M_DICT[el])
        v=p*ph.C_SI/E_tot #units: c (so this is beta)

        # construct true LIS at the same energies (momenta)
        p_ref = ph.p10_DICT[el]
        v_ref = p_ref*ph.C_SI/(10*ph.M_DICT[el] + 931.49*ph.M_DICT[el])
        flux=LIS_norm*((p/p_ref)**(-alpha1))*((v/v_ref)**alpha)
    except:
        return -np.inf
    
    if np.any(flux<=0): return -np.inf
    
    return flux

def flux_brpl_IS(LIS_params, p, el):
    LIS_norm, alpha1, alpha2, pc_br, delta = LIS_params
    p=np.array(p)

    try:

        # construct true LIS at the same energies (momenta)
        # p_br=ph.E_to_p(E_br, ph.M_DICT[el])
        p_br=pc_br/ph.C_SI # in MeV/(m/s)
        p_ref = ph.p10_DICT[el]
        flux=((p/p_br)**(alpha1/delta) + (p/p_br)**(alpha2/delta))
        flux=flux/((p_ref/p_br)**(alpha1/delta) + (p_ref/p_br)**(alpha2/delta))
        
        # LIS_norm is c/4pi n_ref as in Strong 2015 but at 10 GeV/n as in USINE units. Must scale units to match USINE ones
        flux = LIS_norm*(flux**(-delta))

    except:
        return -np.inf
    
    if np.any(flux<=0): return -np.inf
    
    return flux
    
def flux_dbrpl_IS(LIS_params, p, el):
    LIS_norm, alpha1, alpha3, pc_br2, delta2, alpha2, pc_br1, delta1 = LIS_params
    # independent among elements: LIS_norm, alpha1(HE), alpha3(LE), pc_br2(M-L), delta2(M-L)
    # common among elements: alpha2(ME), pc_br1(H-M), delta1(H-M)
    p=np.array(p)

    try:

        # construct true LIS at the same energies (momenta)
        p_br1=pc_br1/ph.C_SI # in MeV/(m/s)
        p_ref = ph.p10_DICT[el]
        flux=((p/p_br1)**(alpha1/delta1) + (p/p_br1)**(alpha2/delta1))
        flux=flux/((p_ref/p_br1)**(alpha1/delta1) + (p_ref/p_br1)**(alpha2/delta1))
        
        # LIS_norm is c/4pi n_ref as in Strong 2015 but at 10 GeV/n as in USINE units. Must scale units to match USINE ones
        flux = LIS_norm*(flux**(-delta1))

        # introduce low-energy cutoff
        p_br2=pc_br2/ph.C_SI # in MeV/(m/s)
        factor=1./((p/p_br2)**(alpha3/delta2) + 1)
        factor = factor**(delta2)
        flux = flux*factor


    except:
        return -np.inf
    
    if np.any(flux<=0): return -np.inf
    
    return flux
    
