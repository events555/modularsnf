import copy

class MatrixOps:
    def __init__(self, ring):
        self.ring = ring

    def mat_mul(self, A, B):
        rows_A = len(A)
        cols_A = len(A[0])
        rows_B = len(B)
        cols_B = len(B[0])
        
        if cols_A != rows_B:
            raise ValueError("Dimension mismatch")
            
        C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                sum_val = 0
                for k in range(cols_A):
                    sum_val = self.ring.add(sum_val, self.ring.mul(A[i][k], B[k][j]))
                C[i][j] = sum_val
        return C

    def identity(self, n):
        return [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    def transpose(self, A):
        return [list(row) for row in zip(*A)]

    def lemma_3_1_transform(self, A):
        """
        Implements Lemma 3.1.
        Input: Matrix A (n x m).
        Output: (U, T) where U is unimodular, T is echelon, and UA = T.
        
        Note: The paper describes a recursive block breakdown (Fig 3.1).
        For the purpose of verifying the math of Triang/Shift first,
        we implement the iterative version described in code fragment
        which is O(nmr) but functional for unit verification.
        """
        n = len(A)
        m = len(A[0])
        U = self.identity(n)
        T = copy.deepcopy(A)
        
        # Implementing the iterative approach from Chapter 3 intro for stability
        r = 0 # current pivot row index (0-based)
        
        for k in range(m): # For each column
            if r >= n: break
            
            # We need to eliminate entries below T[r, k]
            # The paper iterates i from r+1 to n.
            # However, in a PIR, we use Gcdex to eliminate.
            
            # First, ensure the pivot itself gathers the GCD of the column
            for i in range(r + 1, n):
                if T[i][k] == 0:
                    continue
                
                # Apply Gcdex to T[r, k] and T[i, k]
                g, s, t, u, v = self.ring.gcdex(T[r][k], T[i][k])
                
                # Construct the 2x2 transform block
                # [[s, t], [u, v]]
                
                # Update T (Apply to rows r and i)
                # We optimize by only updating relevant rows
                row_r = T[r]
                row_i = T[i]
                
                new_row_r = [self.ring.add(self.ring.mul(s, x), self.ring.mul(t, y)) 
                             for x, y in zip(row_r, row_i)]
                new_row_i = [self.ring.add(self.ring.mul(u, x), self.ring.mul(v, y)) 
                             for x, y in zip(row_r, row_i)]
                
                T[r] = new_row_r
                T[i] = new_row_i
                
                # Update U (Apply to rows r and i of U) to track the transform
                row_Ur = U[r]
                row_Ui = U[i]
                
                new_row_Ur = [self.ring.add(self.ring.mul(s, x), self.ring.mul(t, y)) 
                              for x, y in zip(row_Ur, row_Ui)]
                new_row_Ui = [self.ring.add(self.ring.mul(u, x), self.ring.mul(v, y)) 
                              for x, y in zip(row_Ur, row_Ui)]
                
                U[r] = new_row_Ur
                U[i] = new_row_Ui

            if T[r][k] != 0:
                r += 1
                
        return U, T