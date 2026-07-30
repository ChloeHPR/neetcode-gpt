import numpy as np
from numpy.typing import NDArray


class Solution:
    
    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        self.z=z
        value=1/(1+np.exp(-self.z))
        return np.round(value, 5)

    
    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        self.z=z
        new_z=[]
        for i in z:
          new_z.append(max(0.0,i))
        return new_z

res=Solution()
res.relu([-1,0,1]) 
  
