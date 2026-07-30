import numpy as np
from numpy.typing import NDArray

        
class Solution:

    def binary_cross_entropy(self,y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        self.y_true=y_true
        self.y_pred=y_pred
        epsilon=1e-7
        y_pred=y_pred+epsilon
        n=len(y_true)
        return round((-1/n)*np.sum((y_true*np.log(y_pred))+((1-y_true)*(np.log(1-y_pred)))), 4)
        

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], 
                                   y_pred: NDArray[np.float64]) -> float:
        
        self.y_true=y_true
        self.y_pred=y_pred
        epsilon = 1e-7
        n=len(y_true)
        y_pred=y_pred+epsilon
        
        return round((-1/n)*np.sum(np.sum((y_true*np.log(y_pred)))), 4)
        





sol = Solution()
y_true = np.array([1, 0, 1])
y_pred = np.array([0.9, 0.1, 0.8])

print(y_true.shape)
print(sol.categorical_cross_entropy(y_true, y_pred))