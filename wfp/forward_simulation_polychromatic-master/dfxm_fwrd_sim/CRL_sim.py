import numpy as np
import matplotlib.pyplot as plt 
import tqdm

def find_focal_distance(d1, lens_description):
    """ Find the distance to the focal plane for a given lens and a given d1.

    Parameters:  d1 (float): Distance from sample plane to the first component in the lens_description:
        lens_description (list): Each item in the list should be a dict, that descripes one of several optical components: Lens box,
        free-space propagation, absorbing aperture (which has no effect on the focal length)

    Returns: d2 (float): Distance to the focal plane from the first component in lens_description
            a (float): extra factor of magnification du to thick-lens behaviour
    """

    #Initialize
    a = 1
    R = d1
    L = 0


    # Loop through optical components
    for component in lens_description:

        if component['kind'] == 'free space':
            R_old = R
            R = R + component['length']
            a = a * R / R_old
            L = L + component['length']

        elif component['kind'] == 'lens box':

            # Loop through number of lenslets
            for _ in range(component['N']):

                #Propagate to lenslet
                R_old = R
                R = R + component['T']
                a = a * R / R_old
                L = L + component['T']

                # Multiply by lenslet phases
                R = R*component['f']/(component['f']-R)

        elif component['kind'] not in ['square aperture', 'circular aperture', 'aberration_function', 'aberration_array']:
            print('''Component of kind: "%s" is not known.''' %(component['kind']) )

    # Go back to first component
    R_old = R
    R = R -L
    a = a * R / R_old

    return -R, a


def NIST_Reader(params, datafile_directory = 'XRM_Fourier_ptychography/attenuation_datafiles'):
    """ Reads absorption and refraction data from a datafile

    Parameters: params (dict): Contains the two keys "Material" either the string "Be" or "Al". And "lmbd" the wavelength in mm.
    """

    specifier = params['Material']
    # open file and skip header lines
    fn = datafile_directory + '/' + specifier + '.dat'
    fp = open(fn, 'r')
    for _ in range(3):
        next(fp)

    # Initialize lists
    energy = []
    att_coeff = []
    edges = []
    edge_indexes = [0]


    # Loop through lines
    line = fp.readline() 
    vals = line.split('|')
    energy.append(float(vals[0])) # WARNING!!! HARD CODED DATA POSITION
    att_coeff.append(float(vals[-2])) # WARNING!!! HARD CODED DATA POSITION
    edges.append(float(vals[0])) # WARNING!!! HARD CODED DATA POSITION

    for line in fp:
        if line:

            vals = line.split('|')

            # Keep track of absorption edges
            if float(vals[0]) == energy[-1]:
                edges.append(energy[-1])
                edge_indexes.append(len(energy))

            energy.append(float(vals[0])) # WARNING!!! HARD CODED DATA POSITION
            att_coeff.append(float(vals[-2])) # WARNING!!! HARD CODED DATA POSITION


    fp.close()

    edges.append(energy[-1])
    edge_indexes.append(len(energy))

    # Find data range
    query_energy = 1.2398e-9/params['lmbd']
    index = [query_energy < energ for energ in edges].index(True)
    energy = energy[edge_indexes[index-1]:edge_indexes[index]]
    att_coeff = att_coeff[edge_indexes[index-1]:edge_indexes[index]]

    # Do loglog spline interpolation
    from scipy.interpolate import interp1d
    interpolator = interp1d(np.log(energy), np.log(att_coeff))
    mass_att = np.exp(interpolator(np.log(query_energy)))

    # Read basic data
    fn = datafile_directory + '/basic_data.dat'
    fp = open(fn, 'r')

    for line in fp:
        vals = line.split(' ')
        if vals[0] == specifier:
            dens = float(vals[2])
            atomnumber = float(vals[1])
            mass = float(vals[3])
    fp.close()

    # Calculate beta (dimensionless)
    mu = mass_att * dens * 1e-1 # cm to mm
    beta = params['lmbd'] * mu / 4 / np.pi

    params['beta'] = beta

    # Calculate delta
    electron_density = atomnumber * dens / mass * 6.022e20 # pr mm cubed
    delta = electron_density * params['lmbd']**2 / 2 / np.pi * 2.8179e-12 
    
    params['delta'] = delta
    return beta, delta


def draw_lens(d1, lens_description):
    """ Make a nice little plot of lens components

    Parameters:  lens_description (list): Each item in the list should be a dict, that descripes one of several optical components: Lens box,
        free-space propagation, absorbing aperture (which has no effect on the focal length)

    """

    scaling = 2e3
    n = 20
    y = np.linspace(-1,1,n)
        
    
    

    #Initialize
    a = 1
    R = d1
    L = 0

    plt.plot(np.zeros(n), 1.2*y, 'k-')


    # Loop through optical components
    for component in lens_description:

        if component['kind'] == 'free space':
            plt.plot(np.ones(n)*L, y, 'k--')
            R_old = R
            R = R + component['length']
            a = a * R / R_old
            L = L + component['length']

            plt.plot(np.ones(n)*L, y, 'k--')
        elif component['kind'] == 'lens box':

            # Loop through number of lenslets
            for _ in range(component['N']):

                #Propagate to lenslet
                R_old = R
                R = R + component['T']
                a = a * R / R_old
                L = L + component['T']

                # Multiply by lenslet phases
                R = R*component['f']/(component['f']-R)

                # PLot
                plt.plot(L-y**2*scaling / R , y,'c')

        elif component['kind'] == 'square aperture':

            plt.plot([L,L], [-1, -0.2], 'k', linewidth = 6)
            plt.plot([L,L], [0.2, 1], 'k', linewidth = 6)

        elif component['kind'] not in ['circular aperture', 'aberration_function', 'aberration_array']:
            print('''Component of kind: "%s" is not known.''' %(component['kind']) )

    plt.plot(np.ones(n)*L, y, 'k--')

    # Go back to first component
    R_old = R
    R = R -L
    a = a * R / R_old


    plt.xlabel('x (mm)')
    plt.title('Sketch of lens geometry')
    plt.yticks([])


    plt.show()
    return -R, a


def CRL_propagator(field, d1, lmbd, del_x, lens_description):
    """ FFT-propagator for CRL simulation

    """

    # 
    shape = field.shape

    # Build coordinate arrays sample plane
    x = (np.arange(shape[0])-shape[0]/2)*del_x[0]
    y = (np.arange(shape[1])-shape[1]/2)*del_x[1]

    x = x[:, np.newaxis]
    y = y[np.newaxis, :]

    qx = np.fft.fftfreq(shape[0])/del_x[0]
    qy = np.fft.fftfreq(shape[1])/del_x[1]
    qx = qx[:, np.newaxis]
    qy = qy[np.newaxis, :]

    ### Propagate to lens central plane
    xsq = x**2 + y**2
    nearfield_phase_factors = np.exp(1j*np.pi*xsq/lmbd/d1)


    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(nearfield_phase_factors * field))) # shifted tp base band for neatness

    #Plane space coordinate arrays
    x_lens = np.fft.fftshift(qx)*lmbd*d1
    y_lens = np.fft.fftshift(qy)*lmbd*d1
    lens_rsq = x_lens**2 + y_lens**2

    qx_lens = np.fft.fftshift(x)/lmbd/d1 
    qy_lens = np.fft.fftshift(y)/lmbd/d1
    qsq = qx_lens**2 + qy_lens**2

    #Initialize extra stuff
    a = 1
    R = d1
    L = 0

    # Loop through optical components
    for component in lens_description:

        if component['kind'] == 'free space':

            # Popagate field
            z = component['length']
            prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
            field = np.fft.fft2(prop * np.fft.ifft2(field))

            # take care of extra parameters
            R_old = R
            R = R + component['length']
            a = a * R / R_old
            L = L + component['length']


        elif component['kind'] == 'lens box':

            # Loop through number of lenslets
            for _ in range(component['N']):

                ##Propagate to lenslet
                # Popagate field
                z = component['T']
                prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
                field = np.fft.fft2(prop * np.fft.ifft2(field))
                    
                # take care of extra parameters
                R_old = R
                R = R + component['T']
                a = a * R / R_old
                L = L + component['T']

                # Add lenslet phases
                R = R*component['f']/(component['f']-R)

        elif component['kind'] == 'square aperture':

            # 
            trans = np.logical_and(np.abs(x_lens) < component['width']/2, np.abs(y_lens) < component['width']/2)
            field = field * trans


    ## Go back to first component
    # Popagate field
    z = -L
    prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
    field = np.fft.fft2(prop * np.fft.ifft2(field))

    R_old = R
    R = R - L
    a = a * R / R_old

    # Propagate to focal plane
    field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))

    return field


def CRL_propagator_sheared_grid(field, d1, lmbd, FOV_cen, M, lens_description, lens_angle = [0,0], lens_pos = [0,0], energ_error = 0):
    """ FFT-propagator for CRL simulation. This one works on a non-ortogonal grid defined by the matrix M.

        Params: 
            Field (complex np arrays). Complex envelope of the electric field to be propagated.
            M (2 by 2 real arary). Step sizes of the grid in mm
            lens_description (list): List of optical components in the CRL. The components themselves are dicts

    """
    
    ############# MAKE ARRAYS OF SQUARED COORDINATES IN REAL AND RECIP SPACE  ##########
    # Read out the grid size
    shape = field.shape
    # Sample space coordinates in imaging frame
    xm = M[0,0] * np.arange(shape[0])[:,np.newaxis] + M[0,1]*np.arange(shape[1])[np.newaxis,:]
    ym = M[1,0] * np.arange(shape[0])[:,np.newaxis] + M[1,1]*np.arange(shape[1])[np.newaxis,:]
    # Corresponding recip_space units
    M_inv = np.linalg.inv(M).transpose()
    qxm = M_inv[0,0] * np.fft.fftshift(np.fft.fftfreq(shape[0]))[:,np.newaxis] + M_inv[0,1] * np.fft.fftshift(np.fft.fftfreq(shape[1]))[np.newaxis,:]
    qym = M_inv[1,0] * np.fft.fftshift(np.fft.fftfreq(shape[0]))[:,np.newaxis] + M_inv[1,1] * np.fft.fftshift(np.fft.fftfreq(shape[1]))[np.newaxis,:]

    # Calculate lens plane real space coordinate array:
    # This line is useful if we would give FOV center in different units
    # lens_center = [FOV_cen[0] * M[0,0] + FOV_cen[1] * M[0,1], FOV_cen[0] * M[1,0] + FOV_cen[1] * M[1,1] ]
    lens_x = qxm*d1*lmbd
    lens_y = qym*d1*lmbd
    rsq = lens_x**2+lens_y**2

    # Calculate lens plane recip space coordinate array:
    lens_qx = xm/d1/lmbd
    lens_qx = np.fft.ifftshift(lens_qx)
    lens_qx = lens_qx - lens_qx[0,0]
    lens_qy = ym/d1/lmbd
    lens_qy = np.fft.ifftshift(lens_qy)
    lens_qy = lens_qy - lens_qy[0,0]
    qsq = lens_qx**2+lens_qy**2

    #######   PROPAGATE TO LENS REFERENCE PLANE   #######
    # Nearfield phase factor
    sample_rsq = (xm - FOV_cen[0])**2+(ym - FOV_cen[1])**2
    nearfield_phase_correction = np.exp(1j*np.pi/d1/lmbd*sample_rsq)

    field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(nearfield_phase_correction * field))) # shifted to base band for general neatness, it has zero effect on the output

    #######   CASCADED LENSES #########
    #Initialize extra stuff
    a = 1
    R = d1
    L = 0

    # Loop through optical components
    for component in lens_description:

        print(component)
        if component['kind'] == 'free space':

            # Popagate field
            z = component['length']
            prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
            field = np.fft.fft2(prop * np.fft.ifft2(field))

            # take care of extra parameters
            R_old = R
            R = R + component['length']
            a = a * R / R_old
            L = L + component['length']


        elif component['kind'] == 'lens box':

            # Loop through number of lenslets
            for _ in tqdm.tqdm(range(component['N'])):

                ##Propagate to lenslet
                # Popagate field
                z = component['T']
                prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
                field = np.fft.fft2(prop * np.fft.ifft2(field))
                    
                # take care of extra parameters
                R_old = R
                R = R + component['T']
                a = a * R / R_old
                L = L + component['T']



                # Add lenslet phases
                R = R*component['f']/(component['f']-R)
                # Chromatic aberration
                field = field * np.exp(-1j*np.pi*rsq/lmbd/component['f']*(2*energ_error))
                # Misalignment aberrration
                # CRL rotation and translation
                component_x = lens_pos[0] + L * lens_angle[0]
                component_y = lens_pos[1] + L * lens_angle[1]
                field = field * np.exp(-1j*2*np.pi/lmbd/component['f']*lens_x*component_x) * np.exp(-1j*2*np.pi/lmbd/component['f']*lens_y*component_y)
                # Lens absorption
                cmpnt_sq = (lens_x*a- component_x)**2 + (lens_y*a- component_y)**2
                field = field * np.exp(-cmpnt_sq/2/component['sig_a']**2)



        elif component['kind'] == 'square aperture':

            # CRL rotation and translation
            component_x = lens_pos[0] + L * lens_angle[0]
            component_y = lens_pos[1] + L * lens_angle[1]

            trans = np.logical_and(np.abs(lens_x*a- component_x) < component['width']/2, np.abs(lens_y*a - component_y) < component['width']/2)
            field = field * trans

        elif component['kind'] == 'aberration_function':
            
            # CRL rotation and translation
            component_x = lens_pos[0] + L * lens_angle[0]
            component_y = lens_pos[1] + L * lens_angle[1]
            
            # local coordinate arrays
            x_local = lens_x*a- component_x
            y_local = lens_y*a- component_y

            # Multiply by complec transmission function
            trans = component['function'](x_local,y_local)
            field = field * trans

        elif component['kind'] == 'aberration_array':
            
            # Multiply by complec transmission function
            
            trans = component['array']
            field = field * trans
            
        else:
            print('Component type not understood')


    ## Go back to first component
    # Popagate field
    z = -L
    prop = np.exp(-1j*np.pi*qsq/a**2*lmbd*R*z/(R+z))
    field = np.fft.fft2(prop * np.fft.ifft2(field))

    R_old = R
    R = R - L
    a = a * R / R_old

    # Propagate to focal plane
    field = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))

    return field


def determine_focus_condition(full_optical_length, lens_description, a, b):

    # Find safe interval to search in 
    fN_recip = 0
    L = 0 
    for component in lens_description:
        if component['kind'] == 'lens box':
            fN_recip += component['N'] / component['f'] 
            L += component['N']*component['T']
            
        elif component['kind'] == 'free space':
            L += component['length'] 

    fun = lambda d1 : d1 + find_focal_distance(d1, lens_description)[0]-full_optical_length

    
    # find focus condition
    from scipy.optimize import brentq
    d1 = brentq(fun, a, b)

    return d1


def calculate_abber_array(shape, d1, lmbd, FOV_cen, M, lens_description, lens_angle = [0,0], lens_pos = [0,0]):
    """ Get effective aberrations arrays form functions. Usefull to save a bit of caculation time and to make a pickle-able lens

    """
    
    out = []
    ############# MAKE ARRAYS OF SQUARED COORDINATES IN REAL AND RECIP SPACE  ##########

    # Sample space coordinates in imaging frame
    xm = M[0,0] * np.arange(shape[0])[:,np.newaxis] + M[0,1]*np.arange(shape[1])[np.newaxis,:]
    ym = M[1,0] * np.arange(shape[0])[:,np.newaxis] + M[1,1]*np.arange(shape[1])[np.newaxis,:]
    # Corresponding recip_space units
    M_inv = np.linalg.inv(M).transpose()
    qxm = M_inv[0,0] * np.fft.fftshift(np.fft.fftfreq(shape[0]))[:,np.newaxis] + M_inv[0,1] * np.fft.fftshift(np.fft.fftfreq(shape[1]))[np.newaxis,:]
    qym = M_inv[1,0] * np.fft.fftshift(np.fft.fftfreq(shape[0]))[:,np.newaxis] + M_inv[1,1] * np.fft.fftshift(np.fft.fftfreq(shape[1]))[np.newaxis,:]

    # Calculate lens plane real space coordinate array:
    # This line is useful if we would give FOV center in different units
    # lens_center = [FOV_cen[0] * M[0,0] + FOV_cen[1] * M[0,1], FOV_cen[0] * M[1,0] + FOV_cen[1] * M[1,1] ]
    lens_x = qxm*d1*lmbd
    lens_y = qym*d1*lmbd
    rsq = lens_x**2+lens_y**2

    #######   CASCADED LENSES #########
    #Initialize extra stuff
    a = 1
    R = d1
    L = 0

    # Loop through optical components
    for component in lens_description:

        print(component)
        if component['kind'] == 'free space':

            z = component['length']

            # take care of extra parameters
            R_old = R
            R = R + component['length']
            a = a * R / R_old
            L = L + component['length']


        elif component['kind'] == 'lens box':

            # Loop through number of lenslets
            for _ in range(component['N']):

                ##Propagate to lenslet
                # Popagate field
                z = component['T']

                    
                # take care of extra parameters
                R_old = R
                R = R + component['T']
                a = a * R / R_old
                L = L + component['T']

                # Add lenslet phases
                R = R*component['f']/(component['f']-R)
            


        elif component['kind'] == 'aberration_function':
            
            # CRL rotation and translation
            component_x = lens_pos[0] + L * lens_angle[0]
            component_y = lens_pos[1] + L * lens_angle[1]
            
            # local coordinate arrays
            x_local = lens_x*a- component_x
            y_local = lens_y*a- component_y

            # Multiply by complec transmission function
            trans = component['function'](x_local,y_local)

            out.append( trans)
    return out