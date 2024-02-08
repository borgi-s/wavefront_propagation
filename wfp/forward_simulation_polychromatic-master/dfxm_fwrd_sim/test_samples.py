import numpy as np
import matplotlib.pyplot as plt
import sys

class ScatteringFunction:
    """ Object-oriented version of the sample, which does not save a large array, but calculated the scattering function when needed.
    This is the parent class. It's just a perfect crystal
    """

    def __init__(self, params):
        """ Load the sample and scatting geometry and calculate the coordinate arrays
        """
        self.shape = np.array(params['Geometry']['grid_shape'])
        self.del_x = np.array(params['Geometry']['step_sizes'])

        self.Q = params['Geometry']['hkl'] # Q in hkl-basis

        # Crystal basis in lab frame in physical units
        self.A = np.stack((params['Material']['a'], params['Material']['b'], params['Material']['c'])).transpose()
        self.B =   2*np.pi * np.linalg.inv(self.A).transpose() # Recip lattice basis in lab frame physical units
        self.Q_phys =np.dot(self.B, self.Q) # Q lab frame phys. units
        print(self.Q_phys)

        # Coordinate_arrays. Sample space.
        x = np.arange(self.shape[0])*self.del_x[0]
        y = np.arange(self.shape[1])*self.del_x[1]

        self.x = x[:, np.newaxis]*np.ones(self.shape[:-1]) ## Hacky way to make a 1D matrix that behaves as a 2D matrix
        self.y = y[np.newaxis, :]*np.ones(self.shape[:-1])

        self.has_displacenemt_field_quenstionmark = False


    def __getitem__(self, key):
        """
        VERY hacky indexing. I assume that the frst two indexes are ':', and that the last is an integer aka.: key = (slice(None), slice(None), z_index)
        """


        if self.has_displacenemt_field_quenstionmark:
            zi = key[-1] #YOLO
            z = zi*self.del_x[2]*np.ones(self.shape[:-1]) # lab frame 'z' coord
            r = np.stack((self.x,self.y,z), axis = -1) # lab frame coord vectors with insane indexing convention
            u_array = self.u(r)
        else:
            u_array = np.zeros((*self.x.shape,3))

        return u_array[:,:,0]**2+u_array[:,:,1]**2+u_array[:,:,2]**2#np.dot(u_array,self.Q_phys) / np.linalg.norm(self.Q_phys) ##WARNING DEBUGGING !!!!!!!!!!!!


class EdgeDislocation(ScatteringFunction):
    """ Edge dislocation:
    Edge dislocations are typically described like  ½⟨110⟩{111}, where ⟨110⟩ is the faimily of possible bruges vectors, and {111} is the family of slip planes
    Once we define the Bruges vector, we must choose a slip plane orthogonal to this, i.e. if the edge dislocation is [110], the slip plane will one of four possible +/-(-11+/-1)
    The line direction of the edge dislocation will be brugersXplane, in our example t =  +/-[1-1+/-2] (also known as L)

    center: a poit on the dislocation line in relative lab-frame units, w/e that means to you.
    b: burger's vector in hlk basis
    t: line direction in hkl basis, has to be normal to b in phys. units, easy to check in cubic, less so in other coor. systems


    """

    def __init__(self, ex, center, b = 0.5*np.array([1,1,0]), t = np.array([-1,1,2]) ):
        super().__init__(ex)



        ### GEOMETRY ###

        b = np.dot(self.A,b)
        self.bnorm = np.linalg.norm(b)
        b = b/np.linalg.norm(b) # Normalized Burger's vector

        t = np.dot(self.A,t)
        t = t/np.linalg.norm(t) # Normalized line direction

        # Build rotation matrix from lab to disloc coord system
        n = np.cross(b,t)
        self.Rot = np.stack((n,t,b)).T # Dislocation coord. system unit vectors in crystal basis
        self.Rot_inv = np.linalg.inv(self.Rot) # Is this the same as the transpose?

        self.has_displacenemt_field_quenstionmark = True

        centre_pos = self.shape*self.del_x*center
        self.rotation_center = centre_pos[np.newaxis, np.newaxis, :] # Lab frame phys untis

    def u(self, r):
        r = r-self.rotation_center
        r = np.dot(r, self.Rot)
        pois = 0.334
        alpha = 1e-20

        un = self.bnorm/2/np.pi * ((1-2*pois)/4/(1-pois)*np.log(r[...,2]**2 + r[...,1]**2 + alpha) \
                                + (r[...,2] **2 -  r[...,1]**2) / (2*(1-pois)*(r[...,2]**2 - r[...,1]**2 + alpha)))
        ut = np.zeros(self.shape[:-1])
        ub = self.bnorm/2/np.pi * (np.arctan2(r[...,2], r[...,1]) \
                                + r[...,2] * r[...,1] / (2*(1-pois)*(r[...,2]**2 + r[...,1]**2 + alpha)) )

        u = np.stack((un,ut,ub), axis = -1)
        return np.dot(u, self.Rot_inv)

class PlanarDefect(ScatteringFunction):
    def __init__(self, params, center, plane_norm, disp_vector ):
        super().__init__(params)

        self.u_const = np.dot(self.A, disp_vector)

        print(np.dot(self.u_const, self.Q_phys))
        self.norm = plane_norm
        self.has_displacenemt_field_quenstionmark = True

        centre_pos = self.shape*self.del_x*center
        self.center = centre_pos[np.newaxis, np.newaxis, :] # Lab frame phys untis

    def u(self, r):
        r = r-self.center
        domain = np.dot(r, self.norm) > 0
        return domain[:,:,np.newaxis]*self.u_const[np.newaxis, np.newaxis,:]

class HomogeneousStrain(ScatteringFunction):
    def __init__(self, ex, disp_grad_tensor ):
        super().__init__(ex)

        self.H = disp_grad_tensor
        self.has_displacenemt_field_quenstionmark = True


    def u(self, r):
        return np.dot(r, self.H),

class EdgeDislocationWall(ScatteringFunction):
    def __init__(self, params, cen, U_disloc, bnorm, disloc_spacing, disloc_number, pois = 0.334):
        """
            INPUTS:
                params: geometric information
                cen: position of central disloc in relative units
                U_disloc: rotation matric going from sim. frame to disloc. relative frame
                bnorm: magnitude of Burger's vector in mm
                disloc_spacing: nearest distance from disloc to neighbor in mm
                disloc_number: number of dislocs. I havent written expections for single disloc yet, but I should
                pois: poisson ratio
        """
        super().__init__(params)

        # Store inputs
        self.Rot = U_disloc
        self.Rot_inv = np.linalg.inv(self.Rot).T # Is this the same as the transpose?
        self.has_displacenemt_field_quenstionmark = True
        self.disloc_N = disloc_number
        self.pois = pois
        self.bnorm = bnorm

        # Build list of relative positions of dislocation cores:
        self.distance_along_n = np.linspace( -(disloc_number-1) / 2*disloc_spacing, (disloc_number-1) / 2*disloc_spacing, disloc_number)
        # distance_along_b = np.zeros(disloc_number)


        # Center is in relative coordinates for some dumb reason
        centre_pos = self.shape*self.del_x*np.array(cen)
        self.rotation_center = centre_pos[np.newaxis, np.newaxis, :]
        print(centre_pos, self.rotation_center, self.shape, self.del_x, np.array(cen))

    def u(self, r):

        # plt.imshow(r[:,:,0], aspect = 1/10)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(r[:,:,1], aspect = 1/10)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(r[:,:,2], aspect = 1/10)
        # plt.colorbar()
        # plt.show()

        print(r, 'r')
        # Get coorinates in dislocation frame
        r = r-self.rotation_center
        r = np.dot(r, self.Rot)

        print(r.shape, 'rshape')
        # plt.imshow(r[:,:,0], aspect = 1/10)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(r[:,:,1], aspect = 1/10)
        # plt.colorbar()
        # plt.show()

        # plt.imshow(r[:,:,2], aspect = 1/10)
        # plt.colorbar()
        # plt.show()


        # to avoid div by 0 error
        alpha = 1e-20

        # Initialize
        un = np.zeros(r.shape[:-1])
        ut = np.zeros(r.shape[:-1])
        ub = np.zeros(r.shape[:-1])

        # loop through dislocations
        for ii in range(self.disloc_N):

            r_local = np.array(r)
            r_local[...,1] = r_local[...,1] - self.distance_along_n[ii]

            un += self.bnorm/np.pi/2 * ((1-2*self.pois)/4/(1-self.pois)*np.log(r_local[...,0]**2 + r_local[...,1]**2 + alpha) \
                                                + (r_local[...,0]**2 - r_local[...,1]**2) / (2*(1-self.pois)*(r_local[...,0]**2 - r_local[...,1]**2 + alpha)))
            ub += self.bnorm/np.pi/2 * (np.arctan2(r_local[...,1], r_local[...,0]) \
                                                + r_local[...,0] * r_local[...,1] / (2*(1-self.pois)*(r_local[...,0]**2 + r_local[...,1]**2 + alpha)) )

        # stack
        u = np.stack((ub,un,ut), axis = -1)
        return np.dot(u, self.Rot_inv)
