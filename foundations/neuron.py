import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        self.x=x
        self.w=w
        self.b=b
        #self.activation=activation
        z=np.dot(x,w)+b
        if activation== "sigmoid" :
            return round((1 / (1 + np.exp(-z))),5)
        else:
            return round(max(0.0,z),5)
    
       
res=Solution()
x = [1.0, 2.0]
w = [0.5, 0.5]
b = 0.0
activation = "relu"
res.forward(x,w,b,"sigmoid")

        
        
