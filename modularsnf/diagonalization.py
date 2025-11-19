from modularsnf.matrix import MatrixOps

class DiagonalReduction:
    def __init__(self, ring):
        self.ring = ring
        self.ops = MatrixOps(ring)

    def scalar_merge(self, a, b):
        """
        Implements Lemma 7.10.
        Merges two scalars a, b into g, vb such that g = gcd(a,b) and g | vb.
        
        Returns: 2x2 matrices U, V such that:
        U * diag(a,b) * V = diag(g, vb)
        """
        if a == 0 and b == 0:
            return self.ops.identity(2), self.ops.identity(2)
        
        # 1. Compute Gcdex to get g and the left transform
        # [s t] [a] = [g]
        # [u v] [b]   [0]
        g, s, t, u, v = self.ring.gcdex(a, b)
        
        # U is the matrix from Gcdex
        U = [
            [s, t],
            [u, v]
        ]
        
        # 2. Compute q for the right transform V
        # q = -Div(tb, g)
        # Note: tb is divisible by g because g = sa + tb. 
        # If R is a PID, this is exact. In Z/N, we use our div method.
        tb = self.ring.mul(t, b)
        neg_tb = self.ring.sub(0, tb)
        q = self.ring.div(neg_tb, g)
                
        one = 1
        one_plus_q = self.ring.add(one, q)
        
        V = [
            [1, q],
            [1, one_plus_q]
        ]
        
        return U, V

    def merge_blocks(self, A, B):
        """
        Implements Theorem 7.11.
        """
        n_a = len(A)
        n_b = len(B)
        n = n_a + n_b
        
        # Base Case for Empty (Prevents infinite recursion from size=0 splits)
        if n == 0:
            return [], [], []

        # Base Case: 1x1 matrices (Scalars)
        if n_a == 1 and n_b == 1:
            a = A[0][0]
            b = B[0][0]
            U_local, V_local = self.scalar_merge(a, b)
            
            D = self.ops.create_diagonal([a, b])
            UD = self.ops.mat_mul(U_local, D)
            S = self.ops.mat_mul(UD, V_local)
            return S, U_local, V_local

        # Recursive Step
        t = n_a // 2 
        
        # Work Matrix
        W = self.ops.create_diagonal([A[i][i] for i in range(n_a)] + [B[i][i] for i in range(n_b)])
        U_total = self.ops.identity(n)
        V_total = self.ops.identity(n)

        def apply_sub_merge(blk1_idx, blk2_idx, size):
            # [FIX] Guard against zero-size merges
            if size == 0:
                return

            nonlocal W, U_total, V_total
            
            start1 = blk1_idx * size
            start2 = blk2_idx * size
            
            # Extract diagonals
            D1 = [[0]*size for _ in range(size)]
            D2 = [[0]*size for _ in range(size)]
            for k in range(size):
                D1[k][k] = W[start1+k][start1+k]
                D2[k][k] = W[start2+k][start2+k]

            # RECURSIVE CALL
            S_sub, U_sub, V_sub = self.merge_blocks(D1, D2)
            
            # Update W
            for k in range(size):
                W[start1+k][start1+k] = S_sub[k][k]
                W[start2+k][start2+k] = S_sub[size+k][size+k]

            # Embed Transforms
            U_lifted = self.ops.identity(n)
            V_lifted = self.ops.identity(n)
            
            indices = [start1 + k for k in range(size)] + [start2 + k for k in range(size)]
            
            for r_local in range(2*size):
                for c_local in range(2*size):
                    r_global = indices[r_local]
                    c_global = indices[c_local]
                    U_lifted[r_global][c_global] = U_sub[r_local][c_local]
                    V_lifted[r_global][c_global] = V_sub[r_local][c_local]
            
            U_total = self.ops.mat_mul(U_lifted, U_total)
            V_total = self.ops.mat_mul(V_total, V_lifted)

        # Execute 5 steps
        # If t=0, these will simply return immediately, preventing the loop.
        apply_sub_merge(0, 2, t)
        apply_sub_merge(1, 3, t)
        apply_sub_merge(1, 2, t)
        apply_sub_merge(2, 3, t)
        
        return W, U_total, V_total

    def reduce_diagonal(self, D):
        """
        Main entry point for Proposition 7.7.
        Input: Diagonal Matrix D.
        Output: S (Smith Form), U, V.
        """
        n = len(D)
        if n == 1:
            return D, self.ops.identity(1), self.ops.identity(1)
            
        # Split
        mid = n // 2
        D1 = [row[:mid] for row in D[:mid]]
        D2 = [row[mid:] for row in D[mid:]]
        
        # Recurse on halves
        S1, U1, V1 = self.reduce_diagonal(D1)
        S2, U2, V2 = self.reduce_diagonal(D2)
        
        # Construct pre-merge matrices
        # U_pre = diag(U1, U2)
        U_pre = self.ops.identity(n)
        V_pre = self.ops.identity(n)
        self.ops.embed_block(U_pre, U1, 0, 0)
        self.ops.embed_block(U_pre, U2, mid, mid)
        self.ops.embed_block(V_pre, V1, 0, 0)
        self.ops.embed_block(V_pre, V2, mid, mid)
        
        # Merge
        S, U_merge, V_merge = self.merge_blocks(S1, S2)
        
        # Combine transforms
        U = self.ops.mat_mul(U_merge, U_pre)
        V = self.ops.mat_mul(V_pre, V_merge)
        
        return S, U, V