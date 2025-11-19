# Smith Normal form of Integer matrices mod N (Storjohann)

This is a Python module that implements the algorithm found in Storjohann's PhD dissertation on Algorithms for Matrix Canonical Forms.

It implements all the Lemmas and subsequent subroutines that are necessary.

It validates it by against SymPy using a known equivalence between calculating the Smith Normal form of an integer matrix, and then taking mod N, compared to solving it natively in the ring.