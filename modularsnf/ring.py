class RingZModN:
    def __init__(self, N):
        self.N = N

    def add(self, a, b):
        return (a + b) % self.N

    def sub(self, a, b):
        return (a - b) % self.N

    def mul(self, a, b):
        return (a * b) % self.N

    def div(self, a, b):
        # Warning: Only used when division is exact
        # We use simple modular inverse if coprime, or search if not.
        # For unit testing exact division:
        for x in range(self.N):
            if (b * x) % self.N == (a % self.N):
                return x
        raise ValueError(f"Exact division {a}/{b} impossible in Z/{self.N}")

    def gcdex_primitive(self, a, b):
        """
        Computes extended GCD over integers to simulate
        the Gcdex operation in Z/N.
        Returns (g, s, t) such that s*a + t*b = g
        """
        r0, r1 = a, b
        s0, s1 = 1, 0
        t0, t1 = 0, 1
        
        while r1 != 0:
            q = r0 // r1
            r0, r1 = r1, r0 - q * r1
            s0, s1 = s1, s0 - q * s1
            t0, t1 = t1, t0 - q * t1
            
        return r0, s0, t0

    def gcdex(self, a, b):
        """
        Implements Gcdex(a, b) [cite: 299-301]
        Returns g, s, t, u, v such that:
        [[s, t], [u, v]] * [a, b]^T = [g, 0]^T
        and sv - tu is a unit.
        """
        # We treat inputs as integers for the Euclidean step
        a_int = a % self.N
        b_int = b % self.N
        
        g, s, t = self.gcdex_primitive(a_int, b_int)

        # If b is divisible by a, s=v=1, t=0.
        # However, standard EGCD gives us s, t.
        # We need u, v to complete the unimodular matrix.
        # We choose u = -b/g, v = a/g to zero out the second row.
        
        if g == 0:
             # Case 0, 0
             return 0, 1, 0, 0, 1
             
        # Safety check for exact division (guaranteed by property of GCD)
        u = -(b_int // g)
        v = (a_int // g)
        
        # Reduce all modulo N
        return (g % self.N, s % self.N, t % self.N, u % self.N, v % self.N)