from modularsnf.matrix import MatrixOps


class BandReduction:
    def __init__(self, ring):
        self.ops = MatrixOps(ring)
        self.ring = ring

    def is_upper_b_banded(self, A, b):
        rows = len(A)
        cols = len(A[0])
        for i in range(rows):
            for j in range(cols):
                # Condition 1: j < i (Lower triangular part must be 0)
                if j < i and A[i][j] != 0:
                    return False
                # Condition 2: j >= i + b (Outside the band must be 0)
                if j >= i + b and A[i][j] != 0:
                    return False
        return True

    def triang(self, B, s1, s2):
        """
        Implements Lemma 7.3.
        B is n1 x n1.
        s1 = floor(b/2)
        B2 is the block at rows 0..s1-1, cols s1..s1+s2-1
        """
        n1 = len(B)
        
        # Extract B2
        # B2 rows: 0 to s1 (exclusive)
        # B2 cols: s1 to s1 + s2 (exclusive)
        B2 = [row[s1:s1+s2] for row in B[:s1]]
        
        # We need to triangularize B2^T via left multiplication.
        # Lemma 3.1 gives U such that U * A = Echelon.
        # We want B2 * W = LowerTriangular.
        # (B2 * W)^T = W^T * B2^T = UpperEchelon.
        # So apply Lemma 3.1 to B2^T.
        
        B2_T = self.ops.transpose(B2)
        U_prime, _ = self.ops.lemma_3_1_transform(B2_T)
        
        # The transform V in the paper is diag(I_s1, W) where W = U_prime^T
        W = self.ops.transpose(U_prime)
        
        # Construct full transform V (size n1 x n1)
        # Structure: [ I_s1   0 ]
        #            [ 0      W ]
        # Note: W might be smaller than the remaining space if s2 < remaining.
        # Lemma 7.3 says V acts on the columns corresponding to B2 and B3.
        # The block structure implies W is s2 x s2.
        
        V = self.ops.identity(n1)
        
        # Embed W into V starting at s1, s1
        w_rows = len(W)
        w_cols = len(W[0])
        
        for i in range(w_rows):
            for j in range(w_cols):
                if s1+i < n1 and s1+j < n1:
                    V[s1+i][s1+j] = W[i][j]
                    
        # Apply transform B' = B * V
        B_prime = self.ops.mat_mul(B, V)
        return B_prime, V

    def shift(self, C, s2):
        """
        Implements Lemma 7.4.
        C is n2 x n2.
        C1 is top-left s2 x s2 block.
        C2 is top-right s2 x s2 block.
        """
        # 1. Triangularize C1^T (column ops on C1)
        C1 = [row[:s2] for row in C[:s2]]
        C1_T = self.ops.transpose(C1)
        U_prime_T, _ = self.ops.lemma_3_1_transform(C1_T)
        # U corresponds to U in [cite: 1223]
        U = self.ops.transpose(U_prime_T) # Wait, if U' * C1^T = T, then C1 * U'^T = T^T (lower)
        # The paper says "principal transform U^T such that C1^T U^T is lower triangular" 
        # If U_prime * C1_T = Echelon (Upper), then (U_prime * C1_T)^T = C1 * U_prime^T = Lower.
        # So the matrix applied to the LEFT of C1 is actually U_prime (transposed? no).
        
        # Let's re-read carefully: "compute ... U^T such that C1^T U^T is lower triangular".
        # My Lemma 3.1 computes L such that L * A = Upper.
        # Let A = C1^T. L * C1^T = Upper.
        # (L * C1^T)^T = C1 * L^T = Lower.
        # So the transform applied to the RIGHT of C1 (columns) is L^T.
        # But the formula [cite: 1223] applies U to the LEFT of C.
        # "U^T such that C1^T U^T is lower" -> U acts on rows of C1? No, U^T acts on columns of C1^T.
        # This implies U acts on ROWS of C1. 
        # Let's assume the standard: Use Lemma 3.1 on C1 (cols) to find U such that U*C1 is Upper.
        # Then apply U.
        
        # RE-READING: "compute ... principal transform U^T such that C1^T U^T is lower triangular"
        # If X * Y is Lower, then Y^T * X^T is Upper.
        # So U * C1 is Upper.
        # So U is simply the result of lemma_3_1_transform(C1).
        
        U_local, _ = self.ops.lemma_3_1_transform(C1)
        
        # Embed U_local into full U (n2 x n2)
        U_full = self.ops.identity(len(C))
        for i in range(len(U_local)):
            for j in range(len(U_local[0])):
                U_full[i][j] = U_local[i][j]
        
        # Apply U_full * C
        C_step1 = self.ops.mat_mul(U_full, C)
        
        # 2. Transform V such that (U C2) V is lower triangular.
        # Let temp = (U C2).
        # We want temp * V = Lower.
        # (temp * V)^T = V^T * temp^T = Upper.
        # So we apply Lemma 3.1 to temp^T to get V^T.
        
        C2_step1 = [row[s2:] for row in C_step1[:s2]]
        temp_T = self.ops.transpose(C2_step1)
        V_trans, _ = self.ops.lemma_3_1_transform(temp_T)
        V_local = self.ops.transpose(V_trans)
        
        # Embed V_local into full V (n2 x n2)
        V_full = self.ops.identity(len(C))
        offset = s2
        for i in range(len(V_local)):
            for j in range(len(V_local[0])):
                if offset+i < len(C) and offset+j < len(C):
                    V_full[offset+i][offset+j] = V_local[i][j]
                    
        # Final C' [cite: 1223]
        C_prime = self.ops.mat_mul(C_step1, V_full)
        
        return C_prime, U_full, V_full