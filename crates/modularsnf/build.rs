// Force-link OpenBLAS when the `blas` feature is enabled.
//
// The `openblas-src` crate with the `system` feature emits the correct
// cargo:rustc-link-search and cargo:rustc-link-lib directives for the
// library crate itself, but test/bench binaries sometimes miss them.
// This build script ensures the link flags are always present.

fn main() {
    #[cfg(feature = "blas")]
    {
        println!("cargo:rustc-link-lib=openblas");
    }
}
