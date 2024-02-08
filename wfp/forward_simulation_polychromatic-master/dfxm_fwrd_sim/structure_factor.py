'''
This code is designed as a supliment to Dans_Diffraction giving acces to a x_ray_anomalous function and additional databases

'''
import Dans_Diffraction
import numpy as np

# so that we can use pkg_resources.read_text(databases, 'file') to read the databases
try:
    import importlib.resources as pkg_resources
except ImportError:
    # Try backported to PY<37 `importlib_resources`.
    import importlib_resources as pkg_resources

from . import databases  # relative-import the *package* containing the templates

############################################################################################################
######################################### f0 and related functions #########################################
############################################################################################################

def get_f0(element, Q, database_name = 'f0_WaasKirf'):
    '''
    getter function for f0 given a database name as input. 
    all databases are taken from http://ftp.esrf.fr/pub/scisoft/xop2.3/DabaxFiles/
    f0_dabax.dat appears to be the most recent default file there. 
    it contains the following note:
        This file contains data for only neutral atoms. For ionic stater other DABAX files (like f0_WaasKirf.dat) could be used. 
    f0_WaasKirf.dat is therefore the default here
        '''
    if   database_name == 'f0_WaasKirf': return get_f0_WaasKirf(element,Q)
    elif database_name == 'f0_CromerMann': return get_f0_CromerMann(element,Q)
    elif database_name == 'f0_xop': return get_f0_xop(element,Q)
    elif database_name == 'f0_InterTables': return get_f0_WaasKirf(element,Q)
    elif database_name == 'f0_ITC': return get_f0_ITC(element,Q)
    
    else:
        raise ValueError("No database of the following kind "+database_name+". accepted databases are: f0_WaasKirf, f0_CromerMann, f0_xop, f0_InterTables, f0_ITC")
    return
    

def parse_f0_WaasKirf():
    '''
    parses 'databases/f0_WaasKirf.dat' 
    input: None
    returns:
        dictionary of values in f0_WaasKirf.dat
    '''
    f0_WaasKirf_dict = {}
    f0_WaasKirf_file = pkg_resources.open_text(databases, 'f0_WaasKirf.dat')
    for line in f0_WaasKirf_file:
        if line[0:2] == '#S':
            element = line.split()[2]
        if not line[0] =='#':
            params = np.zeros(11)
            for i, val in enumerate(line.split()):
                params[i]=float(val)
            f0_WaasKirf_dict[element] = params
    return f0_WaasKirf_dict

def get_f0_WaasKirf(element,Q):
    '''
    Getter funciton for f0 from WaasKirf database
    input:
        element: string detailing ion, i.e.: 'Li1+'
        Q: scattering vector, float Q = 4 pi sin(theta) / lambda
    '''
    # parse database if not loaded
    if not 'f0_WaasKirf_dict' in globals():
        global f0_WaasKirf_dict
        f0_WaasKirf_dict = parse_f0_WaasKirf()
    params = f0_WaasKirf_dict[element]
    # params = [ a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]			    
    # f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]		
    # k = sin(theta) / lambda
    # we have Q = 4 pi sin(theta) / lambda
    k = 0.25*Q/np.pi
    f0 = params[5]
    for i in range(5):
         f0 += params[i]*np.exp(-params[i+6]*(k**2))
    return f0

def parse_f0_xop():
    '''
    parses 'databases/f0_xop.dat' 
    input: None
    returns:
        dictionary of values in f0_xop.dat
    '''
    f0_xop_dict = {}
    f0_xop_file = pkg_resources.open_text(databases, 'f0_xop.dat')
    for line in f0_xop_file:
        if line[0:2] == '#S':
            element = line.split()[2]
        if not line[0] =='#' and not line.strip() == '':
            params = np.zeros(11)
            for i, val in enumerate(line.split()):
                params[i]=float(val)
            f0_xop_dict[element] = params
    return f0_xop_dict

def get_f0_xop(element,Q):
    '''
    Getter funciton for f0 from WaasKirf database
    input:
        element: string detailing ion, i.e.: 'Li1+'
        Q: scattering vector, float Q = 4 pi sin(theta) / lambda
    '''
    # parse database if not loaded
    if not 'f0_xop_dict' in globals():
        global f0_xop_dict
        f0_xop_dict = parse_f0_xop()
    params = f0_xop_dict[element]
    # params = [ a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]			    
    # f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]		
    # k = sin(theta) / lambda
    # we have Q = 4 pi sin(theta) / lambda
    k = 0.25*Q/np.pi
    f0 = params[5]
    for i in range(5):
         f0 += params[i]*np.exp(-params[i+6]*(k**2))
    return f0

def parse_f0_CromerMann():
    '''
    parses 'databases/f0_CromerMann.dat' 
    input: None
    returns:
        dictionary of values in f0_CromerMann.dat
    '''
    f0_CromerMann_dict = {}
    f0_CromerMann_file = pkg_resources.open_text(databases, 'f0_CromerMann.dat')
    for line in f0_CromerMann_file:
        if line[0:2] == '#S':
            element = line.split()[2]
        if not line[0] =='#':
            params = np.zeros(11)
            for i, val in enumerate(line.split()):
                params[i]=float(val)
            f0_CromerMann_dict[element] = params
    return f0_CromerMann_dict

def get_f0_CromerMann(element,Q):
    '''
    Getter funciton for f0 from WaasKirf database
    input:
        element: string detailing ion, i.e.: 'Li1+'
        Q: scattering vector, float Q = 4 pi sin(theta) / lambda
    '''
    # parse database if not loaded
    if not 'f0_CromerMann_dict' in globals():
        global f0_CromerMann_dict
        f0_CromerMann_dict = parse_f0_CromerMann()
    params = f0_CromerMann_dict[element]
    # params = [ a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]			    
    # f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]		
    # k = sin(theta) / lambda
    # we have Q = 4 pi sin(theta) / lambda
    k = 0.25*Q/np.pi
    f0 = params[4]
    for i in range(4):
         f0 += params[i]*np.exp(-params[i+5]*(k**2))
    return f0

def parse_f0_InterTables():
    '''
    parses 'databases/f0_InterTables.dat' 
    input: None
    returns:
        dictionary of values in f0_InterTables.dat
    '''
    f0_InterTables_dict = {}
    f0_InterTables_file = pkg_resources.open_text(databases, 'f0_InterTables.dat')
    for line in f0_InterTables_file:
        if line[0:2] == '#S':
            element = line.split()[2]
        if not line[0] =='#':
            params = np.zeros(11)
            for i, val in enumerate(line.split()):
                params[i]=float(val)
            f0_InterTables_dict[element] = params
    return f0_InterTables_dict

def get_f0_InterTables(element,Q):
    '''
    Getter funciton for f0 from WaasKirf database
    input:
        element: string detailing ion, i.e.: 'Li1+'
        Q: scattering vector, float Q = 4 pi sin(theta) / lambda
    '''
    # parse database if not loaded
    if not 'f0_InterTables_dict' in globals():
        global f0_InterTables_dict
        f0_InterTables_dict = parse_f0_InterTables()
    params = f0_InterTables_dict[element]
    # params = [ a1  a2  a3  a4  a5  c  b1  b2  b3  b4  b5 ]			    
    # f0[k] = c + [SUM a_i*EXP(-b_i*(k^2)) ]		
    # k = sin(theta) / lambda
    # we have Q = 4 pi sin(theta) / lambda
    k = 0.25*Q/np.pi
    f0 = params[5]
    for i in range(5):
         f0 += params[i]*np.exp(-params[i+6]*(k**2))
    return f0

def get_f0_ITC(element,Q):
    '''
    Getter funciton for f0 from International Tables for Crystallography Volume C: 
    Mathematical, physical and chemical tables via dans diffraction
    input:
        element: string detailing ion, i.e.: 'Li1+'
        Q: scattering vector, float Q = 4 pi sin(theta) / lambda
    '''
    f0 = Dans_Diffraction.fc.xray_scattering_factor(element, Q)
    if type(Q) == float:
        return f0[0]
    return f0
############################################################################################################
####################################### f1, f2 and related functions #######################################
############################################################################################################
def get_f1f2(element, energy_kev, database_name = 'f1f2_CFXO'):
    '''
    getter function for f1 given a database name as input. 
    all databases are taken from http://ftp.esrf.fr/pub/scisoft/xop2.3/DabaxFiles/
    some works define the atomic scattering factor as:
    (I):   f = f0(Q) +f1(E) +if2(E)
    others as:
    (II):  f = f0(Q) +f1(E) -Z +if2(E)
    we here stick to (I), and this forces a correnction of -Z for certain databases. 
    see for example  p207 of
    "Henke BL Gullikson EM Davis JC X ray interactions: photoabsorption scattering transmission and reflection at E 50 30000 eV Z 1 92 Atomic Data and Nuclear Data Tables July 1993 vol 54 no 2"
        '''
    if   database_name == 'f1f2_BrennanCowan':   return get_f1f2_BrennanCowan(element,energy_kev)
    elif database_name == 'f1f2_asf_Kissel':     return get_f1f2_asf_Kissel(element,energy_kev)
    elif database_name == 'f1f2_Chantler':       return get_f1f2_Chantler(element,energy_kev)
    elif database_name==  'f1f2_CromerLiberman': return get_f1f2_CromerLiberman(element,energy_kev)
    elif database_name == 'f1f2_EPDL97':         return get_f1f2_EPDL97(element,energy_kev)
    elif database_name == 'f1f2_Henke':          return get_f1f2_Henke(element,energy_kev)
    elif database_name == 'f1f2_Sasaki':         return get_f1f2_Sasaki(element,energy_kev)
    elif database_name == 'f1f2_Windt':          return get_f1f2_Windt(element,energy_kev)
    elif database_name == 'f1f2_CFXO':           return get_f1f2_CFXO(element,energy_kev)
    else:
        raise ValueError("No database of the following kind "+database_name+". accepted databases are: f1f2_BrennanCowan, f1f2_asf_Kissel, f1f2_Chantler, f1f2_CromerLiberman, f1f2_EPDL97, f1f2_Henke, f1f2_Sasaki, f1f2_Windt, f1f2_CFXO")
    return


def parse_f1f2_database(filename):
    '''
    parses 'databases/filename' 
    input: 
        filename
    returns:
        dictionary of np.array of values for each element in filename
    '''
    f1f2_dict = {}
    f1f2_file = pkg_resources.open_text(databases, filename)
    element = 'None'
    params = []
    for line in f1f2_file:
        if line[0:2] == '#S':
            f1f2_dict[element] = np.array(params)
            element = line.split()[2]
            params = []
        if not line[0] =='#':
            params.append([])
            for j, val in enumerate(line.split()):
                params[-1].append(float(val))
            
    return f1f2_dict

def get_f1f2_from_params(element, energy_kev, params):
    '''
    
    input: 
        element: i.e. 'Li'
        energy_kev: float or list of floats
        params: 2d np.array of form [[energy, f1, f2], ...]  
    returns:
        f1: float or list of floats
        f2: float or list of floats
    '''
    E = energy_kev*1000
    index = np.searchsorted(params[:,0],E)
    if type(energy_kev) == float :
        if index==len(params): index-=1
    else:
        index[index==len(params)] = len(params)-1
    f1 = params[index-1,1]+(E-params[index-1,0])*(params[index,1]-params[index-1,1])/(params[index,0]-params[index-1,0]) # interpolate f1
    f2 = params[index-1,2]+(E-params[index-1,0])*(params[index,2]-params[index-1,2])/(params[index,0]-params[index-1,0]) # interpolate f2
    return f1, f2
    '''else:
        f1 = np.zeros(len(energy_kev))
        f2 = np.zeros(len(energy_kev))
        for i, en in enumerate(energy_kev):
            E = en*1000
            index = np.searchsorted(params[:,0],E)
            if index==len(params): index-=1
            f1[i] = params[index-1,1]+(E-params[index-1,0])*(params[index,1]-params[index-1,1])/(params[index,0]-params[index-1,0]) # interpolate f1
            f2[i] = params[index-1,2]+(E-params[index-1,0])*(params[index,2]-params[index-1,2])/(params[index,0]-params[index-1,0]) # interpolate f2
        return f1, f2'''

def get_f1f2_asf_Kissel(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_asf_Kissel_dict' in globals():
        global f1f2_asf_Kissel_dict
        f1f2_asf_Kissel_dict = parse_f1f2_database('f1f2_asf_Kissel.dat')
    params = f1f2_asf_Kissel_dict[element][:,[0,4,5]]
    params = np.copy(params)
    params[:,0]*=1000 # this database used kev, not ev. convert to ev to conform to the others
    f1, f2 = get_f1f2_from_params(element, energy_kev, params)
    return f1-get_Z(element), f2
    
def get_f1f2_BrennanCowan(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_BrennanCowan_dict' in globals():
        global f1f2_BrennanCowan_dict
        f1f2_BrennanCowan_dict = parse_f1f2_database('f1f2_BrennanCowan.dat')
    params = f1f2_BrennanCowan_dict[element]
    return get_f1f2_from_params(element, energy_kev, params)
    
def get_f1f2_Chantler(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_Chantler_dict' in globals():
        global f1f2_Chantler_dict
        f1f2_Chantler_dict = parse_f1f2_database('f1f2_Chantler.dat')
    params = f1f2_Chantler_dict[element]
    params = np.copy(params)
    params[:,0]*=1000 # this database used kev, not ev. convert to ev to conform to the others
    f1, f2 = get_f1f2_from_params(element, energy_kev, params)
    return f1-get_Z(element), f2
    
def get_f1f2_CromerLiberman(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_CromerLiberman_dict' in globals():
        global f1f2_CromerLiberman_dict
        f1f2_CromerLiberman_dict = parse_f1f2_database('f1f2_CromerLiberman.dat')
    params = f1f2_CromerLiberman_dict[element]
    return get_f1f2_from_params(element, energy_kev, params)
    
def get_f1f2_EPDL97(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_EPDL97_dict' in globals():
        global f1f2_EPDL97_dict
        f1f2_EPDL97_dict = parse_f1f2_database('f1f2_EPDL97.dat')
    params = f1f2_EPDL97_dict[element]
    f1, f2 = get_f1f2_from_params(element, energy_kev, params)
    return f1-get_Z(element), f2
    
def get_f1f2_Henke(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_Henke_dict' in globals():
        global f1f2_Henke_dict
        f1f2_Henke_dict = parse_f1f2_database('f1f2_Henke.dat')
    params = f1f2_Henke_dict[element]
    f1, f2 = get_f1f2_from_params(element, energy_kev, params)
    return f1-get_Z(element), f2

def get_f1f2_Sasaki(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_Sasaki_dict' in globals():
        global f1f2_Sasaki_dict
        f1f2_Sasaki_dict = parse_f1f2_database('f1f2_Sasaki.dat')
    params = f1f2_Sasaki_dict[element]
    return get_f1f2_from_params(element, energy_kev, params)
    
def get_f1f2_Windt(element,energy_kev):
    # parse database if not loaded
    if not 'f1f2_Windt_dict' in globals():
        global f1f2_Windt_dict
        f1f2_Windt_dict = parse_f1f2_database('f1f2_Windt.dat')
    params = f1f2_Windt_dict[element]
    f1, f2 = get_f1f2_from_params(element, energy_kev, params)
    return f1-get_Z(element), f2

def get_f1f2_CFXO(element,energy_kev):
    f1, f2 = Dans_Diffraction.fc.atomic_scattering_factor(element, energy_kev)
    if type(energy_kev) == float:
        return f1[0]-get_Z(element), f2[0]
    return f1-get_Z(element), f2


############################################################################################################
####################################### crystal structure factor #######################################
############################################################################################################

# anomalous scattering function for dan's diffractoin
def x_ray_anomalous(Scatter, HKL, energy_kev=None, ionize_fn = None, f0_database = 'f0_WaasKirf', f1f2_database = 'f1f2_BrennanCowan', return_chi = True):
        """
        Calculate the structure factor for the given HKL including anomalous terms, using x-ray scattering factors
        input: 
            scatterer: Dans_Diffraction.classes_scattering.Scattering object
            HLK: list or list of lists(i.e: [1,0,0] or [[1,0,0],[2,0,0],[3,0,0]] )
            energy_kev: default None, energy in kev used for anomalous terms. If None, use config from the scatter object
            ionize_fn: optional function that accepts an element and potentially returns an ion i.e. some variant on:
                    def ionize_fn(atom):
                        if atom == 'Li': return 'Li1+'
                        if atom == 'Nb': return 'Nb5+'
                        if atom == 'O': return 'O2-'
                        return atom
            f0_database: the f0 database to use, see get_f0 for alternatives
            f1f2_database: the f1 and f2 database to use, see get_f1f2
            return_chi: default True, toggle for wether to return chi, or (F, or intensity) 
        returns:
            an array with the same length as HKL, giving the real intensity at each reflection.
            or 
            if Scatter._return_structure_factor is True, return the scatter function
        """
        if energy_kev is None:
            energy_kev = Scatter._energy_kev
        if ionize_fn is None:
            ionize_fn = Scatter.ionize_fn
        HKL = np.asarray(np.rint(HKL),dtype=float).reshape([-1,3])
        Nref = len(HKL)
        
        uvw,atom_types,label,occ,uiso,mxmymz = Scatter.xtl.Structure.get()
        Nat = len(uvw)
        
        Qmag = Scatter.xtl.Cell.Qmag(HKL) # magnitude of q vector
        atom_form_factors = np.zeros(len(atom_types), dtype = complex)
        for i, atom in enumerate(atom_types):
            # get xray scattering factor (f0)
            #atom_form_factors[i] += Dans_Diffraction.fc.xray_scattering_factor(atom, Qmag) # Get comlpex atomic_scattering_factor from dans diffraction (i.e. https://henke.lbl.gov/optical_constants/asf.html)
            if type(ionize_fn) == type(None):
                atom_form_factors[i] += get_f0(atom, Qmag, database_name = f0_database)
            else:
                atom_form_factors[i] += get_f0(ionize_fn(atom), Qmag, database_name = f0_database)
            # add complex atomic scattering factor
            #asf1, asf2 = Dans_Diffraction.fc.atomic_scattering_factor(atom, energy_kev)
            asf1, asf2 = get_f1f2(atom, energy_kev, database_name = f1f2_database)
            atom_form_factors[i] += complex(asf1, asf2)
        ff = atom_form_factors
        
        # Get Debye-Waller factor
        if Scatter._use_isotropic_thermal_factor:
            dw = functions_crystallography.debyewaller(uiso, Qmag)
        elif Scatter._use_anisotropic_thermal_factor:
            raise Exception('anisotropic thermal factor calcualtion not implemented yet')
        else:
            dw = 1
        
        # Calculate dot product
        dot_KR = np.dot(HKL, uvw.T)
        # Calculate structure factor
        # Broadcasting used on 2D ff
        SF =  np.sum(ff*dw*occ*np.exp(-1j*2*np.pi*dot_KR),axis=1)
        
        SF = SF/Scatter.xtl.scale
        
        # we want to return chi for this project so:
        if return_chi: 
            chi_over_F = get_chi_over_F(Scatter, energy_kev)
            return SF*chi_over_F
        # retain ability to return SF and I for compatibily with the rest of Dans_diffraction
        if Scatter._return_structure_factor: return SF
        # Calculate intensity
        I = SF * np.conj(SF)
        return np.real(I)
    
    
def get_chi_over_F(Scatter, energy_kev=None):
    '''
    Gets the conversion factor to convert from:
        F (scattering ~ number of electrons in unit cell) 
    to:
        chi (scattering per volume)
    Further documentation in section three of MadsDerivesDynamicalScatteringTheory.pdf
    input:
        scatterer: Dans_Diffraction.classes_scattering.Scattering object
        energy_kev: default None. If None, use configuration from the scatter object
    '''
    if energy_kev is None:
        energy_kev = Scatter._energy_kev
    #prepare parameters
    wavelength = 4.135667696*10**-15 *299792458 / energy_kev /1000 # h[eV⋅s]*c[m/s]/energy_kev[kev]/1[1/k] = [m]
    k = 2*np.pi/wavelength # in m
    volume = Scatter.xtl.Cell.volume()*10**-30 # in m^3
    classical_electron_radius = 2.8179403227*10**-15 # in m
    # calculte ratio chi_over_F
    numerator = 4*np.pi*classical_electron_radius # in m
    denominator = k**2*volume # [m**-2]*[m**3] = [m]
    chi_over_F = numerator/denominator # [m]/[m] = [-]
    #return
    return chi_over_F       
    
def setup_dans_diffraction(cif_file, wavelength_in_um, ionize_fn = None):
    '''
    function that configures dans difffraction the way we want it.
    input:
        cif_file: string, cif_file
        wavelength_in_um: float, wavelength in um
        ionize_fn: function that ioanizes atoms, optional. f.eks:
                def ionize_fn(atom):
                    if atom == 'Li': return 'Li1+'
                    if atom == 'Nb': return 'Nb5+'
                    if atom == 'O': return 'O2-'
                    return atom
    return:
        xtl: xtl object from dans_diffraction
    '''
    #import Dans_Diffraction
    Dans_Diffraction.classes_scattering.Scattering.x_ray_anomalous = x_ray_anomalous # assign x_ray_anomalous function to dan's diffraction
    xtl = Dans_Diffraction.Crystal(cif_file)
    xtl.generate_lattice()
    wavelength = wavelength_in_um*10**-6 # in meters
    energy_kev = 4.135667696*10**-15 *299792458 / wavelength /1000 # h[eV⋅s]*c[m⋅s]/wavelength[m]
    xtl.Scatter.setup_scatter(scattering_type='xray', energy_kev=energy_kev)
    xtl.Scatter._use_isotropic_thermal_factor = False
    xtl.Scatter._return_structure_factor = True
    if not type(ionize_fn) == type(None):
        xtl.Scatter.ionize_fn = ionize_fn
    else:
        xtl.Scatter.ionize_fn = None
    return xtl


########################## crystal rotation functions #####################
def rotate(x,z,phi):# 2D rotation function 
    '''
    generic rotation funciton for generating the crystal rotation function 
    '''
    return x*np.cos(phi)-z*np.sin(phi),z*np.cos(phi)+x*np.sin(phi)
def rotate_x(loc,phi):#rotate_x: rotates around x axis (horisontal)
    '''
    in-place rotation around x axis for generating the crystal rotation function 
    '''
    x,z=rotate(loc[1],loc[2],phi)
    loc[1] = x
    loc[2] = z
def rotate_y(loc,phi):# rotate_y: rotates around y axis (vertical)
    '''
    in-place rotation around y axis for generating the crystal rotation function 
    '''
    x,z=rotate(loc[0],loc[2],phi)
    loc[0] = x
    loc[2] = z
def rotate_z(loc,phi):#rotate_z: rotates around z axis (out-of-plane)
    '''
    in-place rotation around z axis for generating the crystal rotation function 
    '''
    x,z=rotate(loc[0],loc[1],phi)
    loc[0] = x
    loc[1] = z
#f1f2_BrennanCowan = pkg_resources.read_text(databases, 'f1f2_BrennanCowan.dat')


Z_table = [[1, 'H', 'Hydrogen'],
    [2, 'He', 'Helium'],
    [3, 'Li', 'Lithium'],
    [4, 'Be', 'Beryllium'],
    [5, 'B', 'Boron'],
    [6, 'C', 'Carbon'],
    [7, 'N', 'Nitrogen'],
    [8, 'O', 'Oxygen'],
    [9, 'F', 'Fluorine'],
    [10, 'Ne', 'Neon'],
    [11, 'Na', 'Sodium'],
    [12, 'Mg', 'Magnesium'],
    [13, 'Al', 'Aluminum'],
    [14, 'Si', 'Silicon'],
    [15, 'P', 'Phosphorus'],
    [16, 'S', 'Sulfur'],
    [17, 'Cl', 'Chlorine'],
    [18, 'Ar', 'Argon'],
    [19, 'K', 'Potassium'],
    [20, 'Ca', 'Calcium'],
    [21, 'Sc', 'Scandium'],
    [22, 'Ti', 'Titanium'],
    [23, 'V', 'Vanadium'],
    [24, 'Cr', 'Chromium'],
    [25, 'Mn', 'Manganese'],
    [26, 'Fe', 'Iron'],
    [27, 'Co', 'Cobalt'],
    [28, 'Ni', 'Nickel'],
    [29, 'Cu', 'Copper'],
    [30, 'Zn', 'Zinc'],
    [31, 'Ga', 'Gallium'],
    [32, 'Ge', 'Germanium'],
    [33, 'As', 'Arsenic'],
    [34, 'Se', 'Selenium'],
    [35, 'Br', 'Bromine'],
    [36, 'Kr', 'Krypton'],
    [37, 'Rb', 'Rubidium'],
    [38, 'Sr', 'Strontium'],
    [39, 'Y', 'Yttrium'],
    [40, 'Zr', 'Zirconium'],
    [41, 'Nb', 'Niobium'],
    [42, 'Mo', 'Molybdenum'],
    [43, 'Tc', 'Technetium'],
    [44, 'Ru', 'Ruthenium'],
    [45, 'Rh', 'Rhodium'],
    [46, 'Pd', 'Palladium'],
    [47, 'Ag', 'Silver'],
    [48, 'Cd', 'Cadmium'],
    [49, 'In', 'Indium'],
    [50, 'Sn', 'Tin'],
    [51, 'Sb', 'Antimony'],
    [52, 'Te', 'Tellurium'],
    [53, 'I', 'Iodine'],
    [54, 'Xe', 'Xenon'],
    [55, 'Cs', 'Cesium'],
    [56, 'Ba', 'Barium'],
    [57, 'La', 'Lanthanum'],
    [58, 'Ce', 'Cerium'],
    [59, 'Pr', 'Praseodymium'],
    [60, 'Nd', 'Neodymium'],
    [61, 'Pm', 'Promethium'],
    [62, 'Sm', 'Samarium'],
    [63, 'Eu', 'Europium'],
    [64, 'Gd', 'Gadolinium'],
    [65, 'Tb', 'Terbium'],
    [66, 'Dy', 'Dysprosium'],
    [67, 'Ho', 'Holmium'],
    [68, 'Er', 'Erbium'],
    [69, 'Tm', 'Thulium'],
    [70, 'Yb', 'Ytterbium'],
    [71, 'Lu', 'Lutetium'],
    [72, 'Hf', 'Hafnium'],
    [73, 'Ta', 'Tantalum'],
    [74, 'W', 'Tungsten'],
    [75, 'Re', 'Rhenium'],
    [76, 'Os', 'Osmium'],
    [77, 'Ir', 'Iridium'],
    [78, 'Pt', 'Platinum'],
    [79, 'Au', 'Gold'],
    [80, 'Hg', 'Mercury'],
    [81, 'Tl', 'Thallium'],
    [82, 'Pb', 'Lead'],
    [83, 'Bi', 'Bismuth'],
    [84, 'Po', 'Polonium'],
    [85, 'At', 'Astatine'],
    [86, 'Rn', 'Radon'],
    [87, 'Fr', 'Francium'],
    [88, 'Ra', 'Radium'],
    [89, 'Ac', 'Actinium'],
    [90, 'Th', 'Thorium'],
    [91, 'Pa', 'Protactinium'],
    [92, 'U', 'Uranium'],
    [93, 'Np', 'Neptunium'],
    [94, 'Pu', 'Plutonium'],
    [95, 'Am', 'Americium'],
    [96, 'Cm', 'Curium'],
    [97, 'Bk', 'Berkelium'],
    [98, 'Cf', 'Californium'],
    [99, 'Es', 'Einsteinium'],
    [100, 'Fm', 'Fermium'],
    [101, 'Md', 'Mendelevium'],
    [102, 'No', 'Nobelium'],
    [103, 'Lr', 'Lawrencium'],
    [104, 'Rf', 'Rutherfordium'],
    [105, 'Db', 'Dubnium'],
    [106, 'Sg', 'Seaborgium'],
    [107, 'Bh', 'Bohrium'],
    [108, 'Hs', 'Hassium'],
    [109, 'Mt', 'Meitnerium'],
    [110, 'Ds', 'Darmstadtium'],
    [111, 'Rg', 'Roentgenium'],
    [112, 'Cn', 'Copernicium'],
    [113, 'Nh', 'Nihonium'],
    [114, 'Fl', 'Flerovium'],
    [115, 'Mc', 'Moscovium'],
    [116, 'Lv', 'Livermorium'],
    [117, 'Ts', 'Tennessine'],
    [118, 'Og', 'Oganesson'],
        ]
Z_dict = {}
for line in Z_table:
    Z_dict[line[1]]=line[0]
def get_Z(element):
    '''get the atomic number of the element'''
    return Z_dict[element]
