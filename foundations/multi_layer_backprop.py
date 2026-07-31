import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,x: List[float], W1: List[List[float]], b1: List[float],W2: List[List[float]], b2: List[float],y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        self.x=np.array(x)
        self.W1=np.array(W1)
        self.b1=b1
        self.W2=np.array(W2)
        self.b2=b2
        self.y_true=y_true
        resultats={}
        z1=np.dot(self.x,self.W1.T)+b1
        A1=np.maximum(z1,0)
        z2=np.dot(A1,self.W2.T)+b2
        y_pred=z2
        L=np.mean((y_pred-y_true)**2)
        
        n = len(y_true)
        dL_dz2 = (2 / n) * (y_pred - y_true)
        db2=dL_dz2
        dW2=np.outer(dL_dz2,A1)
        dL_dA1 = np.dot(dL_dz2, W2)
        dL_dz1=dL_dA1*(z1 > 0)
        db1=dL_dz1
        dW1=np.outer(dL_dz1,x)

        resultats['loss'] = np.round(L,5)
        resultats['dW1'] = np.round(dW1,5)
        resultats['db1'] = np.round(db1,5)
        resultats['dW2'] = np.round(dW2,5)
        resultats['db2'] = np.round(db2,5)
 
        return resultats


res=Solution()
x = [1.0, 2.0]
W1 = [[1.0, 0.0], [0.0, 1.0]]  # 2x2 identity
b1 = [0.0, 0.0]
W2 = [[0.5, 0.5]]              # 1x2
b2 = [0.0]
y_true = [1.0]
res.forward_and_backward(x,W1,b1,W2,b2,y_true)