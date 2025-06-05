# Population-Coding
A take away and expansion from academic project. The job is to train a robotic arm to control its joint angles in order to reach target points within a reach field on a table.
The setup is shown in Fig.

![image](https://github.com/user-attachments/assets/5f8ae15e-f8e8-4c4c-96d0-900776336620)

There are two coordinate systems to represent the reach points:
• Field polar coordinates (r,θ), where r ≥ 0 is the radius, and θ is a normalized angle in the range (−1,1] (instead of (−π,π]), and
• Worldcoordinates (x,y), as shown in Fig. The file robot.py includes the class Arm, which implements this task and its environment.
That file also includes the class Population, which implements a population of neurons. You can look at the code and its documentation
for more information.
The class Decoding performs linear decoding from the activity of a population of neurons. If the activities of N neurons over P samples is stored in a P × N matrix A, and the
desired output corresponding to those P samples is stored in the P × K matrix F, then the optimal linear decoding weights are set by finding the least-squares weights D ∈ RN×K
that solve

