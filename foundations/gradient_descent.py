class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places

        self.iterations=iterations
        self.learning_rate=learning_rate
        self.init=init
        
        self.entrainement(iterations)

        # On renvoie la position finale (self.init) arrondie à 5 décimales pour le test
        return round(self.init, 5)

    def derivates(self):
        return 2*self.init

    def pas(self):
        self.init=self.init-self.learning_rate*self.derivates()

    def entrainement(self, iterations):
        for i in range(iterations):
            self.pas()

ia = Solution()

# 2. On lance l'entraînement avec 10 étapes, un taux de 0.4 en partant de x = 3
resultat = ia.get_minimizer(iterations=10, learning_rate=0.4, init=3)

# 3. On affiche le résultat à l'écran !
print("Le minimum trouvé par notre IA est :", resultat)
