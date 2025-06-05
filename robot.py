import numpy as np
import matplotlib.pyplot as plt
import torch


class Population(torch.nn.Module):
    '''
     pop = Population(n_neurons=50, radius=1., dim=1)

     Creates a Population object for encoding data.

     Inputs:
      n_neurons  number of neurons
      radius     radius of expected inputs
                 Most inputs are expected to be in the range [-radius, radius]
      dim        dimension of inputs

     The radius parameter is used to select the initial weights and biases
     in a way that favours regression fits in the range [-radius, radius].

     Usage:
     > pop = Population(n_neurons=50, radius=1.5, dim=3)
     > x = torch.tensor([[1., -0.6, 1.3], [-1.2, 1.4, -0.2]])
     > A = pop(x)
     > A.shape
      torch.Size([2, 50])

     Note: Inputs outside the radius can be encoded, but the decodings will
     likely be less accurate.
    '''
    def __init__(self, n_neurons=50, radius=0.1, dim=1):
        super(Population, self).__init__()
        self.dim = dim          # dimension of encoding
        self.N = n_neurons      # number of neurons
        self.radius = radius    # radius for training decodings

        self.encoder = torch.nn.Linear(self.dim, self.N, bias=True)
        self.activation = torch.nn.Sigmoid()

        # Set up encoder weights and biases
        gain = torch.rand(size=(self.N,)) * 20. / self.radius
        enc = torch.normal(0, 1, size=(self.N, self.dim))
        for r,alpha in zip(enc, gain):
            r *= alpha / torch.norm(r)
        self.encoder.weight.data = enc
        self.encoder.bias.data = torch.rand(size=(self.N,)) * 2 * self.radius - self.radius
        self.encoder.bias.data *= torch.norm(self.encoder.weight.data, dim=1)


    def forward(self, x):
        '''
         A = pop(x)

         Encodes inputs into activities.

         Input:
          x   (P, dim) tensor holding one input in each row

         Output:
          A   (P, N) tensor holding corresponding activities in rows
        '''
        x = self.encoder(x)
        x = self.activation(x)
        return x
    


class Arm():
    '''
     arm = Arm()

     Creates an Arm object for simulating a monkey's reaching arm.
     
     Usage:
     > arm = Arm()
     > arm.draw_arm([0.5, 0.8])  # Give it shoulder and elbow angles
      (draws arm in current figure)
    '''
    def __init__(self):
        # Parameters of monkey's arm
        self.scale = 1.
        self.R1 = 1.1 * self.scale
        self.R2 = 1 * self.scale
        self.R = self.R1**2 - self.R2**2
        self.fieldcentre = self.scale*np.array([-0.5, 1.])
        self.fieldradius = 0.6 * self.scale


    def draw(self, joint_angs):
        '''
         arm.draw(joint_angs)

         Draws the arm given the joint angles.

         Input:
          joint_angs  (2,) NumPy array of normalized joint angles (in the
                      range [-1,1]) of the form [shoulder, elbow]
        '''
        shoulder, elbow = joint_angs
        v = self.finger_location(joint_angs)
        v1x = self.R1*np.cos(shoulder*np.pi)
        v1y = self.R1*np.sin(shoulder*np.pi)
        plt.plot([0,v1x,v[0]],[0,v1y,v[1]], color='0.7')
        plt.plot([v[0]],[v[1]], '.', color='0.6')
        
        self.draw_field()


    def field2world(self, polar):
        r, theta = polar
        x = r*np.cos(theta*np.pi) + self.fieldcentre[0]
        y = r*np.sin(theta*np.pi) + self.fieldcentre[1]
        return np.array([x, y])
    

    def finger_location(self, joint_angs):
        '''
         v = arm.finger_location(joint_angs)

         Returns the location of the fingertip given the joint angles.

         Input:
          joint_angs  (2,) NumPy array of normalized joint angles (in the
                      range [-1,1]) of the form [shoulder, elbow]

         Output:
          v           (2,) NumPy array of the location of the fingertip
                      in World coordinates [x,y]
        '''
        shoulder, elbow = joint_angs
        v1x = self.R1*np.cos(shoulder*np.pi)
        v1y = self.R1*np.sin(shoulder*np.pi)
        omega = 1 + shoulder - elbow
        v2x = v1x + self.R2*np.cos(omega*np.pi)
        v2y = v1y + self.R2*np.sin(omega*np.pi)
        return np.array([v2x, v2y])


    def draw_field(self):
        '''
         arm.draw_field()

         Draws a circle around the reaching field.
        '''
        th = np.linspace(0, 2*np.pi, 90, endpoint=True)
        borderx = self.fieldradius*np.cos(th) + self.fieldcentre[0]
        bordery = self.fieldradius*np.sin(th) + self.fieldcentre[1]
        plt.plot(borderx, bordery, 'g--')
        plt.axis('scaled')
        plt.axis(self.scale * np.array([-1.5,1,0,2]))
        plt.xlabel('World x')
        plt.ylabel('World y')


    def draw_target_point(self, polar):
        '''
         arm.draw_target_point(polar)

         Draws a red dot at the target point specified by normalized
         polor coordinates.

         Input:
          polar  (2,) NumPy array in Field coords
                 ie. in the form [r, theta], where theta is a
                 normalized angle in the range [-1,1].
        '''
        pt = self.field2world(polar)
        plt.plot(pt[0], pt[1], 'ro')
   

    def random_field_point(self):
        '''
         polar = arm.random_field_point()

         Returns a random point within the reaching field.
         The return value, polar, is a  in Field coordinates.
         
         Output:
          polar  (2,) NumPy array in Field coords
                 ie. in the form [r, theta], where theta is a
                 normalized angle in the range [-1,1].
        '''
        r = np.sqrt(np.random.rand())*self.fieldradius
        theta = np.random.rand()*2 - 1
        pt = self.field2world([r, theta])
        return np.array([r, theta]), pt


    def is_in_field(self, v):
        '''
         bool = arm.is_in_field(v)

         Tests if a point (in World coords) is inside the reaching field.
         True if the point is inside the field, and False otherwise.

         Input:
          v     (2,) NumPy array in World coordinates (x,y)

         Output:
          bool  Boolean (True or False)
        '''
        r = np.sqrt((v[0]-self.fieldcentre[0])**2 + (v[1]-self.fieldcentre[1])**2)
        return r < self.fieldradius


    def select_points(self, P):
        '''
         field, world, joints = arm.select_points(P)

         Selects P random points within the reaching field.

         Input:
          P    number of points to select

         Output:
          field  (P,2) NumPy array of points in Field coords
          world  (P,2) NumPy array of points in World coords
          joints (P,2) NumPy array of joint angles
        '''
        field = np.zeros((P,2))
        world = np.zeros((P,2))
        joints = np.zeros((P,2))
        counter = 0
        while counter < P:
            # Choose random joint angles
            shoulder = np.random.rand()
            elbow = np.random.rand()

            # Find out where the fingertip is
            v = self.finger_location([shoulder, elbow])
            r = np.sqrt((v[0]-self.fieldcentre[0])**2 + (v[1]-self.fieldcentre[1])**2)
            theta = np.arctan2(v[1]-self.fieldcentre[1], v[0]-self.fieldcentre[0])
            polar = np.array([r, theta/np.pi])
            # If it's in the reach field, record the point and joint angles
            if r<self.fieldradius:
                field[counter,:] = polar
                world[counter,:] = v
                joints[counter,:] = [shoulder, elbow]
                counter += 1

        return field, world, joints
