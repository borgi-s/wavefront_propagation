import numpy as np
import pickle
import dfxm_fwrd_sim.CRL_sim as CRL
import dfxm_fwrd_sim.parameter_parser as par
import os
import concurrent.futures

def propagate(n, params, lens_rot, integration_name, propagation_name):

    # Calculate scattered beam vector and some angles
    k0 = params['Beam']['k_0_ref']
    k = np.linalg.norm(k0)
    Q = params['Geometry']['Q']
    kh = k0 + Q
    eta = np.arctan2(kh[1], kh[0])
    alpha_h = np.arccos(kh[2]/k)
    
    # Old frame coordinates
    xo = np.arange(params['Geometry']['grid_shape'][0]) * params['Geometry']['step_sizes'][0]
    yo = np.arange(params['Geometry']['grid_shape'][1]) * params['Geometry']['step_sizes'][1]
    xo, yo = np.meshgrid(xo,yo,indexing = 'ij')

    # imaging frame coordinates
    x = (xo * np.cos(eta) + yo * np.sin(eta)) * np.cos(alpha_h)
    y = (-xo * np.sin(eta) + yo * np.cos(eta))
    
    # pick mode
    modekey = f'Mode {n:d}'

    # Load integrated field
    with open(params['I/O']['root_dir'] + 'integrated_field/' + integration_name + f'/mode_{n:04d}.npy', 'rb') as fid:
        Eh_exit = pickle.load(fid)

    # Load lens
    with open(params['I/O']['obj_lens'], 'rb') as fid:
        obj_lens = pickle.load(fid)

    # Energy error
    energ_error = 1 - params['Beam']['lmbd_ref']/params[modekey]['lmbd']
    
    # Step size matrix in the imaging coord. system
    delx_o = params['Geometry']['step_sizes'][0]; dely_o = params['Geometry']['step_sizes'][0]
    delx_img =np.array([[delx_o*np.cos(eta)*np.cos(alpha_h), dely_o*np.sin(eta)*np.cos(alpha_h)], [-delx_o*np.sin(eta), dely_o*np.cos(eta)]])
    
    # Set rotation factors
    rotation_factors = np.exp(1j*k*(x*lens_rot[0] + y*lens_rot[1]))
    
    # REad FOV center
    FOV_cen = params['Optics']['FOV_cen']
    
    # Propagate
    Eh_det = CRL.CRL_propagator_sheared_grid(Eh_exit*rotation_factors, params['Optics']['d1'], params[modekey]['lmbd'], FOV_cen, delx_img, obj_lens, energ_error = energ_error)

    # Save propagated mode
    root_dir = params['I/O']['root_dir']

    if not os.path.isdir(params['I/O']['root_dir'] + 'integrated_field/' + integration_name + '/propagated_field'):
        os.makedirs(root_dir + 'integrated_field/' + integration_name + '/propagated_field')

    if not os.path.isdir(params['I/O']['root_dir'] + 'integrated_field/' + integration_name + '/propagated_field/' + propagation_name):
        os.makedirs(root_dir + 'integrated_field/' + integration_name + '/propagated_field/' + propagation_name)

    with open(params['I/O']['root_dir'] + 'integrated_field/' + integration_name + '/propagated_field/' + propagation_name + f'/mode_{n:04d}.npy', 'wb+') as fid:
        pickle.dump(Eh_det, fid)

def propagate_parallel(par_fn, lens_rot, integration_name, propagation_name, processes = None, modeslist = None):

    # Read params
    params = par.par_read(par_fn)

    if processes is None:
        processes = params['Status']['processes']
    if modeslist is None:
        modeslist = list(range(params['Beam']['N_modes']))
    
    print(processes)
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        futures_list = []
        for modenumber in modeslist:
            futures_list.append(executor.submit(propagate, modenumber, params, lens_rot, integration_name, propagation_name))

        executor.shutdown(wait=True)
        print(' ')

    return 1