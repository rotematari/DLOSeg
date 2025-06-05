from der.utils import *
import matplotlib.pyplot as plt
import cv2
import numpy as np
"""
this code is an implementation for the 
Differentiable Discrete Elastic Rods for Real-Time Modeling of Deformable Linear Objects
from the paper: 
https://github.com/roahmlab/DEFORM
https://arxiv.org/abs/2406.05931

Algorithm 1 One Step Prediction with DEFORM
Inputs: ˆ Xt, ˆ Vt, ut 
1: θ∗ t(Xt) = argminθt P(Xt,θt,α) ▷ Section.4.1 
2: ˆ Vt+1 = ˆ Vt −∆tM−1∂P(Xt,θ∗ t(Xt,α),α) ∂Xt 
3: ˆ Xt+1 = ˆ Xt +∆t ˆ Vt+1 +DNN ▷ (3) ▷ Section.4.2 
4: Inextensibility Enforcement on ˆ Xt+1 ▷ Section.4.3 
5: ˆ Vt+1 = (ˆ Xt+1 − ˆ Xt)/∆t ▷ Velocity Update return ˆ Xt+1, ˆ Vt+1

"""
def generate_inputs():
    # Generate random inputs for the rod state and velocity
    Xt = np.random.rand(12)  # Current state of the rod (12 parameters)
    Vt = np.random.rand(12)  # Current velocity of the rod (12 parameters)
    ut = np.random.rand(12)  # Poses of the edges of the rod (12 parameters)
    
    return Xt, Vt, ut

def main():
    
    
    # getnerate inputs 
    # Xt is the current state of the rod
    # Vt is the current velocity of the rod
    # ut is the 2 poses of the edges of the rod ut ∈ R12
    
    Xt, Vt, ut = generate_inputs()
    
    # Generate material properties alpha(bending modulus) betta(twisting modulus)
    # 
    
    
    
if __name__ == "__main__":
    main()