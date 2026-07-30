import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        self.z=z
        new_z=[]
        for i in z:
            new_z.append((np.exp(i-np.max(z)))/(np.sum(np.exp(z-np.max(z)))))
        #print(new_z)
        return np.round(new_z, 4)
        


