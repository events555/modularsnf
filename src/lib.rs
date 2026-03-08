mod ring;

use pyo3::prelude::*;

/// Native Rust acceleration for modularsnf.
///
/// Import as `from modularsnf._rust import RustRingZModN`.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ring::RustRingZModN>()?;
    Ok(())
}
