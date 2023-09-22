import numpy as np

def laue(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Laue case for a fixed rocking angle and crystal thickness.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_0 (N by 1 complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by 1 complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j*np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = -1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = -1j*k/2/np.cos(alpha_h)*(chi_0+beta) - 1j*np.tan(alpha_h)*q*2*np.pi

    # Eigenvalues for each decpuples 2x2 problem
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1*t1 - v2*t2)/(v1 - v2) * ff)
    E_h = np.fft.ifft((t1 - t2)/(v1 - v2) * ff)

    return E_0, E_h

def bragg_finite(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Bragg case for a fixed rocking angle and crystal thickness.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_0 (N by 1 complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by 1 complex numpy array): Complex real space amplitudes of scattered beam.
    '''

    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j/np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.sin(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.sin(alpha_h)*(chi_0+beta) + 1j/np.tan(alpha_h)*q*2*np.pi

    # Eigenvalues
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_2 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_1 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v2 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v1 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission of of modes
    t2 = np.exp(eigval_1*L)
    t1 = np.exp(eigval_2*L)

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Calculate reflection and transmission
    E_h = np.fft.ifft((1 -t1 / t2) / (v1+ -t1 / t2*v2) * ff)
    E_0 = np.fft.ifft((v1*t1 -t1 *v2) / (v1-t1 / t2*v2) * ff)

    return E_0, E_h

def bragg_inf(E_init, del_x, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Bragg case for a fixed rocking angle and infinite thickness.
    In this case there is no transmitted beam so it only assigns one output.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_h (N by 1 complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j/np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.sin(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.sin(alpha_h)*(chi_0+beta) + 1j/np.tan(alpha_h)*q*2*np.pi

    # Check which eigenvalue is negative
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    sign = np.sign(np.real(eigval_1))

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Find components of the corresponding eigenvector
    R =  np.fft.ifft(2 * A_h0 / (-A_00 + A_hh + sign * squareroot_term) * ff)

    return R

def bragg_finite_2d_evaluation(E_init, del_x, L, M, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Bragg case for a fixed rocking angle and crystal thickness.
        This function evaluates the solution throughout the crystal, not just on the surfaces, 
        so it can be used to illustate what happens in the interior of the crystal.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        phi (optional, float): rocking angle

    Returns:
        E_0 (N by 1 complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by 1 complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given corresponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j/np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = 1j*k/2/np.sin(alpha_h)*C*chi_h
    A_hh = 1j*k/2/np.sin(alpha_h)*(chi_0+beta) + 1j/np.tan(alpha_h)*q*2*np.pi

    # Eigenvalues
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_2 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_1 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors (normalization says that second component is == 1, WolframAlpha style)
    v2 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v1 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Build depth coordinate axis
    z = np.linspace(0, L, M)

    # Transmission of of modes
    t2 = np.exp(eigval_1[:, np.newaxis]*z[np.newaxis, :])
    t1 = np.exp(eigval_2[:, np.newaxis]*z[np.newaxis, :])

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Relative mode weigth (we choose mode 1 with weigth 1, and normalize in the end)
    m2 = -t1[:,-1] / t2[:,-1]

    # Calculate reflection and transmission
    norm = v1 + m2*v2
    E_0 = np.fft.ifft( (v1[:, np.newaxis]*t1 + m2[:, np.newaxis]*v2[:, np.newaxis]*t2 ) / norm[:, np.newaxis]  * ff[:, np.newaxis], axis = 0)
    E_h = np.fft.ifft( (t1 + m2[:, np.newaxis]*t2 ) / norm[:, np.newaxis] * ff[:, np.newaxis], axis = 0)

    return E_0, E_h


def laue_rockingcurve(E_init, phi, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1):

    ''' Perfect crystal propagator in the Laue case for a fixed crystal thickness as a function of rocking angle.
        The same can be achieved by running 'laue' for different rocking angles, 
        but this one vectorizes over the rocking angle for efficient evaluation.

    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        phi (M by 1, float array): rocking angle
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        
    Returns:
        E_0 (N by M complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by M complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j*np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = -1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = -1j*k/2/np.cos(alpha_h)*(chi_0+beta[np.newaxis,:]) - 1j*np.tan(alpha_h)*q[:, np.newaxis]*2*np.pi

    # Eigenvalues for each decpuples 2x2 problem
    squareroot_term = np.sqrt( A_00[:, np.newaxis]**2 + A_hh**2 - 2*A_00[:, np.newaxis]*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00[:, np.newaxis] + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00[:, np.newaxis] + A_hh )

    # Eigenvectors
    v1 = -(-A_00[:, np.newaxis] + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00[:, np.newaxis] + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1*L)
    t2 = np.exp(eigval_2*L)

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1*t1 - v2*t2)/(v1 - v2) * ff[:, np.newaxis], axis = 0)
    E_h = np.fft.ifft((t1 - t2)/(v1 - v2) * ff[:, np.newaxis], axis = 0)

    return E_0, E_h

def laue_depth_dependece(E_init, del_x, L, lmbd, alpha_0, alpha_h, chi_0, chi_h, chi_hm = None, C = 1, phi = 0):

    ''' Perfect crystal propagator in the Laue case for a fixed rocking angle and variable crystal thickness.
        The same can be achieved by running 'laue' for different crystal thickness, 
        but this one vectorizes over the crystal thickness for efficient evaluation.
    Parameters:
        E_init (N by 1 complex numpy array): Complex real space amplitude of the incident beam.
        phi (M by 1, float array): rocking angle
        del_x (float): Step size in transverse direction.
        L (float): Crystal thickness in logitudinal direction.
        lmbd (float): wavelength in same units as del_x and L
        alpha_0 (float): Angle of incidence of incident beam
        alpha_h (float): Angle of incidence of scattered beam
        chi_0 (complex float): average electric susceptibility
        chi_h (complex float): fourier coeff. of the electric susceptibility corresponding to the reflection
        chi_hm (optional, complex float): fourier coeff. of the electric susceptibility corresponding to the back-reflection
        C (optional, float): Polarization factor
        
    Returns:
        E_0 (N by M complex numpy array): Complex real space amplitudes of transmitted beam.
        E_h (N by M complex numpy array): Complex real space amplitudes of scattered beam.
    '''


    # Buid recip space coordinate arrays. 
    q = np.fft.fftfreq(len(E_init))/del_x # Full period frequency, same unit as input.

    # If chi_hm is not explicitly given, we assume that the chi_h given correcponds to a central reflection
    # and that the imaginary part is the absorption related annommalous part:
    if chi_hm is None:
        chi_hm = chi_h

    # Matrix elements
    k = 2*np.pi/lmbd
    twotheta = alpha_0 - alpha_h
    beta = 2*np.sin(twotheta)*phi
    A_00 = -1j*k/2/np.cos(alpha_0)*chi_0 - 1j*np.tan(alpha_0)*q*2*np.pi
    A_0h = -1j*k/2/np.cos(alpha_0)*C*chi_hm
    A_h0 = -1j*k/2/np.cos(alpha_h)*C*chi_h
    A_hh = -1j*k/2/np.cos(alpha_h)*(chi_0+beta) - 1j*np.tan(alpha_h)*q*2*np.pi

    # Eigenvalues for each decpuples 2x2 problem
    squareroot_term = np.sqrt( A_00**2 + A_hh**2 - 2*A_00*A_hh + 4*A_0h*A_h0 )
    eigval_1 = 0.5*(squareroot_term + A_00 + A_hh )
    eigval_2 = 0.5*(-squareroot_term + A_00 + A_hh )

    # Eigenvectors
    v1 = -(-A_00 + A_hh + squareroot_term) / 2 / A_h0
    v2 = -(-A_00 + A_hh - squareroot_term) / 2 / A_h0

    # Transmission coeff of modes
    t1 = np.exp(eigval_1[:, np.newaxis]*L[np.newaxis,:])
    t2 = np.exp(eigval_2[:, np.newaxis]*L[np.newaxis,:])

    # Transform initial condition
    ff = np.fft.fft(E_init)

    # Transmission and reflection
    E_0 = np.fft.ifft((v1[:, np.newaxis]*t1 - v2[:, np.newaxis]*t2)/(v1[:, np.newaxis] - v2[:, np.newaxis]) * ff[:, np.newaxis], axis = 0)
    E_h = np.fft.ifft((t1 - t2)/(v1[:, np.newaxis] - v2[:, np.newaxis]) * ff[:, np.newaxis], axis = 0)

    return E_0, E_h
