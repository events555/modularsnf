pub mod ring;
mod diagonal;

use pyo3::prelude::*;

/// Native Rust acceleration for modularsnf.
///
/// Import as `from modularsnf._rust import RustRingZModN`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ring::RustRingZModN>()?;
    m.add_function(wrap_pyfunction!(diagonal::rust_smith_from_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal::rust_merge_smith_blocks, m)?)?;
    Ok(())
}
