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
    
    def reduce(self, A, b, t=0):
            """
            Implements Algorithm BandReduction (A, b).
            
            Input: Upper b-banded Matrix A (n x n)
                b > 2
                t (number of last columns known to be zero)
                
            Output: A' (upper floor(b/2)+1 banded), U, V 
                    such that U*A*V = A'.
            """
            n = len(A)
            if b <= 2:
                # Base case for recursion
                return A, self.ops.identity(n), self.ops.identity(n)

            # 1. Parameters
            # s1 defines the step size for the outer loop (Triang)
            # s2 defines the step size for the inner loop (Shift)
            s1 = b // 2
            n1 = (b // 2) + b - 1
            s2 = b - 1
            n2 = 2 * (b - 1)

            # 2. Augmentation
            # "By augmenting A with at most 2b rows and columns... we may assume 
            # that A has at least 2b trailing columns of zeros".
            # We pad with 2*b to safely handle the n1/n2 block sizes near the boundary.
            pad_size = 2 * b - t
            B = self.ops.pad_matrix(A, pad_size)
            n_aug = len(B)

            # Initialize global transforms for this reduction phase.
            # These accumulate the sequence of principal transforms.
            U_accum = self.ops.identity(n_aug)
            V_accum = self.ops.identity(n_aug)

            # 3. Outer Loop
            # Iterates to ceil((n - t) / s1) - 1
            limit_i = (n - t + s1 - 1) // s1  

            for i in range(limit_i):
                # 3a. Apply Triang to subB[i*s1, n1]
                # This clears the "upper triangle" of the current block.
                row_start = i * s1
                
                # Extract the n1 x n1 block starting at diagonal index i*s1
                block_slice = [row[row_start : row_start + n1] for row in B[row_start : row_start + n1]]
                
                # Perform Triang (Lemma 7.3)
                # Returns local transform V_local such that block * V_local has reduced band.
                B_prime_block, V_local = self.triang(block_slice, s1, s2)
                
                # Update the working matrix B
                self.ops.embed_block(B, B_prime_block, row_start, row_start)
                
                # Update Global V accumulator.
                # We lift the n1 x n1 V_local into the global n_aug x n_aug space
                # by embedding it into an identity matrix.
                V_global_step = self.ops.embed_identity(n_aug, V_local, row_start, row_start)
                V_accum = self.ops.mat_mul(V_accum, V_global_step)

                # 3b. Inner Loop (Chasing the bulge)
                # The fill-in created by Triang is chased down the diagonal.
                # Limit: ceil((n - t - (i + 1) * s1) / s2) - 1
                numerator = n - t - (i + 1) * s1
                if numerator > 0:
                    limit_j = (numerator + s2 - 1) // s2
                    
                    for j in range(limit_j):
                        # Apply Shift to subB[(i+1)s1 + j*s2, n2]
                        offset = (i + 1) * s1 + j * s2
                        
                        # Extract the n2 x n2 block
                        block_shift = [row[offset : offset + n2] for row in B[offset : offset + n2]]
                        
                        # Perform Shift (Lemma 7.4)
                        # Returns C' and local U, V such that C' = U * C * V
                        C_prime, U_local, V_local_shift = self.shift(block_shift, s2)
                        
                        # Update working matrix B
                        self.ops.embed_block(B, C_prime, offset, offset)
                        
                        # Update Global U
                        # Note: Shift applies U from the left
                        # U_new = U_local_embedded * U_old
                        U_global_step = self.ops.embed_identity(n_aug, U_local, offset, offset)
                        U_accum = self.ops.mat_mul(U_global_step, U_accum)
                        
                        # Update Global V
                        # V_new = V_old * V_local_embedded
                        V_global_shift = self.ops.embed_identity(n_aug, V_local_shift, offset, offset)
                        V_accum = self.ops.mat_mul(V_accum, V_global_shift)

            # 4. Return sub_B[0, n]
            # We crop the matrix back to its original size n.
            A_prime = [row[:n] for row in B[:n]]
            U_final = [row[:n] for row in U_accum[:n]]
            V_final = [row[:n] for row in V_accum[:n]]

            return A_prime, U_final, V_final