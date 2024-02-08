import numpy as np
import dfxm_fwrd_sim.parameter_parser as par
import pickle
import sys
sys.path.insert(2, '/dtu/3d-imaging-center/projects/2022_QIM_PMP/analysis/borgi/Wavefront propagation/dynamical_diffraction-master/dynamical_diffraction/')
from strained_crystal_3d import laue_exponential_heun_vertical
import os
import concurrent.futures


def integrate_mode_n(n, params_fn, phi, integration_name = None):

    #
    if integration_name is None:
        integration_name = f'phi_{phi*1e6:04.0f}_murad'

    # Read params
    params = par.par_read(params_fn)

    # Select mode
    modekey = f'Mode {n:d}'

    # k space geometry
    k0_ii = params[modekey]['k_0']
    kh_ii = k0_ii + params[f'Geometry']['Q']
    k0_ref = params['Beam']['k_0_ref']


    alpha_0 = np.arctan2(k0_ii[0], k0_ii[2])
    alpha_h = np.arctan2(kh_ii[0], kh_ii[2])

    # Load materials parameters
    chi_0_ii = params[modekey]['chi_0_Re'] + 1j*params[modekey]['chi_0_Im']
    chi_h_ii = params[modekey]['chi_h_Re'] + 1j*params[modekey]['chi_h_Im']

    # Load initial condition
    with open(params[modekey]['incident_field'], 'rb') as fid:
        E_init = pickle.load(fid)

    ### Fudge initial condition (We assume that incident mode is given relative to the nominal
    # direction, which generally differs from the mode direction, so we rotate the initial
    # condition to account for this. The derivation only exists in my head ATM.)
    #Build coordinate arrays
    x = np.arange(params['Geometry']['grid_shape'][0]) * params['Geometry']['step_sizes'][0]
    y = np.arange(params['Geometry']['grid_shape'][1]) * params['Geometry']['step_sizes'][1]
    x, y = np.meshgrid(x,y,indexing = 'ij')
    del_k = k0_ref*np.linalg.norm(k0_ii)/np.linalg.norm(k0_ref) - k0_ii
    E_init = E_init * np.exp(-1j*(x*del_k[0] + y*del_k[1]))

    # Load displacement field
    # Load initial condition
    with open(params['I/O']['displacement_field'], 'rb') as fid:
        u = pickle.load(fid)

    E_0, E_h = laue_exponential_heun_vertical(E_init, u, params['Geometry']['step_sizes'], params['Geometry']['grid_shape'], params[modekey]['lmbd'], alpha_0, alpha_h, chi_0_ii, chi_h_ii, phi = phi)

    # Unfudge scattered beam
    E_h = E_h * np.exp(+1j*(x*del_k[0] + y*del_k[1]))


    root_dir = params['I/O']['root_dir']
    # Save output
    if not os.path.isdir(root_dir + 'integrated_field'):
        os.makedirs(root_dir + 'integrated_field')

    if not os.path.isdir(root_dir + 'integrated_field/'+ integration_name):
        os.makedirs(root_dir + 'integrated_field/'+ integration_name)

    with open(root_dir + 'integrated_field/' + integration_name + f'/mode_{n:04d}.npy', 'wb+') as fid:
        pickle.dump(E_h, fid)
    return 1

def integrate_parallel(params_fn, phi, integration_name = None, processes = None, modeslist = None):

    #
    if integration_name is None:
        integration_name = f'phi_{phi*1e6:04.0f}_murad'

    # Read params
    params = par.par_read(params_fn)

    if processes is None:
        processes = params['Status']['processes']
    if modeslist is None:
        modeslist = list(range(params['Beam']['N_modes']))

    
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        futures_list = []
        for modenumber in modeslist:
            futures_list.append(executor.submit(integrate_mode_n,modenumber, params_fn, phi, integration_name))

        executor.shutdown(wait=True)
        print(' ')

    return 1
