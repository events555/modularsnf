from modularsnf.matrix import MatrixOps
from modularsnf.diagonalization import DiagonalReduction

class BidiagonalToSmith:
    def __init__(self, ring):
        self.ring = ring
        self.ops = MatrixOps(ring)
        self.diag_reducer = DiagonalReduction(ring)

    def _swap_rows(self, M, r1, r2):
        """Swaps rows r1 and r2 in matrix M."""
        M[r1], M[r2] = M[r2], M[r1]

    def _swap_cols(self, M, c1, c2):
        """Swaps cols c1 and c2 in matrix M."""
        for r in range(len(M)):
            M[r][c1], M[r][c2] = M[r][c2], M[r][c1]

    def run(self, A):
        """
        Implements Proposition 7.12: Upper 2-Banded -> Smith Form.
        References: Storjohann Diss. pp. 117-124 [cite: 58, 1386, 1387]
        """
        n = len(A)
        
        # Base Cases
        if n == 0: return [], [], []
        if n == 1: return [list(row) for row in A], self.ops.identity(1), self.ops.identity(1)
            
        # Partitioning parameters [cite: 1398, 1399]
        n1 = (n - 1) // 2
        k_idx = n1 + 1 # Corresponds to 'k' in paper for loop bounds (start of block 2)
        
        # Global Transforms
        U = self.ops.identity(n)
        V = self.ops.identity(n)
        W = [list(row) for row in A]

        # --- Step 1: Clear Super-Diagonals ---
        # Push non-zeros down the diagonal from the "glue" point downwards.
        # [cite: 1400-1413]
        for i in range(k_idx, n): 
            val_diag = W[i][i]
            val_super = W[i-1][i]
            
            if val_super == 0: continue

            # Gcdex on (super, diag)
            g, s, t, u, v = self.ring.gcdex(val_super, val_diag)
            
            # Update W rows (Left Transform)
            row_top = W[i-1]
            row_bot = W[i]
            new_top = [self.ring.add(self.ring.mul(s, x), self.ring.mul(t, y)) for x,y in zip(row_top, row_bot)]
            new_bot = [self.ring.add(self.ring.mul(u, x), self.ring.mul(v, y)) for x,y in zip(row_top, row_bot)]
            W[i-1] = new_top
            W[i] = new_bot
            
            # Update U rows (Left Transform tracks row ops on W)
            row_U_top = U[i-1]
            row_U_bot = U[i]
            new_U_top = [self.ring.add(self.ring.mul(s, x), self.ring.mul(t, y)) for x,y in zip(row_U_top, row_U_bot)]
            new_U_bot = [self.ring.add(self.ring.mul(u, x), self.ring.mul(v, y)) for x,y in zip(row_U_top, row_U_bot)]
            U[i-1] = new_U_top
            U[i] = new_U_bot

        # --- Step 2: Recursive Calls ---
        # Reduce the two diagonal blocks B1 and B2 to SNF. [cite: 1439]
        # Note: B1 ends at n1-1 (inclusive), B2 starts at n1+1.
        # The row/col at 'n1' is the "glue" and is excluded from recursion.
        
        B1 = [row[:n1] for row in W[:n1]]
        B2 = [row[n1+1:] for row in W[n1+1:]]
        
        if B1:
            S1, U1, V1 = self.run(B1)
            self.ops.embed_block(W, S1, 0, 0)
            
            U_embed = self.ops.embed_identity(n, U1, 0, 0)
            V_embed = self.ops.embed_identity(n, V1, 0, 0)
            
            # Update globals
            # Note: U_embed * U (accumulate left), V * V_embed (accumulate right)
            W = self.ops.mat_mul(U_embed, W)
            W = self.ops.mat_mul(W, V_embed)
            U = self.ops.mat_mul(U_embed, U)
            V = self.ops.mat_mul(V, V_embed)

        if B2:
            S2, U2, V2 = self.run(B2)
            self.ops.embed_block(W, S2, n1+1, n1+1)
            
            U_embed = self.ops.embed_identity(n, U2, n1+1, n1+1)
            V_embed = self.ops.embed_identity(n, V2, n1+1, n1+1)
            
            W = self.ops.mat_mul(U_embed, W)
            W = self.ops.mat_mul(W, V_embed)
            U = self.ops.mat_mul(U_embed, U)
            V = self.ops.mat_mul(V, V_embed)

        # --- Step 3: Permute Rows (CRITICAL FIX) ---
        # Move the "glue" row (index n1) to the bottom (index n-1).
        # This ensures the top-left (n-1)x(n-1) block is strictly diagonal. 
        # We accomplish this by bubbling row n1 down to n-1.
        for i in range(n1, n - 1):
            self._swap_rows(W, i, i+1)
            self._swap_rows(U, i, i+1) # Row ops on W => Row ops on U

        # --- Step 4: Reduce Principal (n-1)x(n-1) ---
        # Now that the glue row is at the bottom, the top-left is diagonal.
        # [cite: 1461]
        if n > 1:
            # Extract strict diagonal entries (safe now)
            diag_entries = [W[x][x] for x in range(n-1)]
            D_mat = self.ops.create_diagonal(diag_entries)
            
            # Reduce to SNF
            S_sub, U_sub, V_sub = self.diag_reducer.reduce_diagonal(D_mat)
            
            U_embed = self.ops.embed_identity(n, U_sub, 0, 0)
            V_embed = self.ops.embed_identity(n, V_sub, 0, 0)
            
            W = self.ops.mat_mul(U_embed, W)
            W = self.ops.mat_mul(W, V_embed)
            U = self.ops.mat_mul(U_embed, U)
            V = self.ops.mat_mul(V, V_embed)

        # --- Step 5 & 6: Push Pivot Column to End ---
        # The "glue" column is currently at index n1. Move it to n-1.
        # [cite: 1469]
        for j in range(n1, n - 1):
            self._swap_cols(W, j, j+1)
            self._swap_cols(V, j, j+1) # Col ops on W => Col ops on V

        # At this point, the matrix has the "L" shape:
        # Diagonal block D (size n-1) + Last Row + Last Col.
        
        # --- Step 7: Stab Loop ---
        # Condition the last column using Stab. [cite: 1494]
        col = n - 1
        for i in range(n - 2, -1, -1):
            val_a = W[i][col]     
            val_b = W[i+1][col]   
            val_c = W[i][i]       
            
            if val_b == 0: continue 
            
            c = self.ring.stab(val_a, val_b, val_c)
            
            # Calculate q 
            # q = -Div(c * A[i+1, i+1], A[i, i])
            # Note: Due to swaps, W[i+1][i+1] is the correct diagonal entry below current.
            denom = W[i][i]
            num = self.ring.mul(c, W[i+1][i+1])
            
            if denom == 0:
                q = 0 
            else:
                try:
                    # Using the exact logic from paper
                    q = self.ring.div(self.ring.sub(0, num), denom)
                except ValueError:
                    # Should not happen in a PIR/PID if Stab works, but safe fallback
                    q = 0 
            
            # Row Op: Row[i] += c * Row[i+1] [cite: 1499]
            for cx in range(n):
                term = self.ring.mul(c, W[i+1][cx])
                W[i][cx] = self.ring.add(W[i][cx], term)
                
                term_u = self.ring.mul(c, U[i+1][cx])
                U[i][cx] = self.ring.add(U[i][cx], term_u)
            
            # Col Op: Col[i+1] += q * Col[i] [cite: 1500]
            # Note: This acts on column i+1, which is a diagonal column, NOT the last column.
            for rx in range(n):
                term = self.ring.mul(q, W[rx][i])
                W[rx][i+1] = self.ring.add(W[rx][i+1], term)
                
                term_v = self.ring.mul(q, V[rx][i])
                V[rx][i+1] = self.ring.add(V[rx][i+1], term_v)

        # --- Step 8: Clear Last Column ---
        # Use Gcdex to clear the last column against the diagonals. [cite: 1510]
        for i in range(n - 1):
            val_diag = W[i][i]
            val_col = W[i][n-1]
            
            if val_col == 0: continue
            
            # Gcdex on row i (diagonal) and row i (col n-1)
            g, s, t, u, v = self.ring.gcdex(val_diag, val_col)
            
            # Right multiply by [[s, u], [t, v]] acting on cols i and n-1
            # W' = W * M. V' = V * M.
            
            # We must update the whole column i and n-1 for W and V
            for r in range(n):
                # Update W
                w_i = W[r][i]
                w_last = W[r][n-1]
                W[r][i]   = self.ring.add(self.ring.mul(s, w_i), self.ring.mul(t, w_last))
                W[r][n-1] = self.ring.add(self.ring.mul(u, w_i), self.ring.mul(v, w_last))
                
                # Update V
                v_i = V[r][i]
                v_last = V[r][n-1]
                V[r][i]   = self.ring.add(self.ring.mul(s, v_i), self.ring.mul(t, v_last))
                V[r][n-1] = self.ring.add(self.ring.mul(u, v_i), self.ring.mul(v, v_last))

        # --- Step 9: Final Cleanup ---
        # The matrix is now diagonal. Reduce it one last time to ensure divisibility chain.
        D_final = [W[i][i] for i in range(n)]
        D_mat = self.ops.create_diagonal(D_final)
        
        S_final, U_final, V_final = self.diag_reducer.reduce_diagonal(D_mat)
        
        U = self.ops.mat_mul(U_final, U)
        V = self.ops.mat_mul(V, V_final)
        
        return S_final, U, V