pub mod ring;
pub mod diagonal;
pub mod echelon;
pub mod band;
pub mod snf;

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;

/// Native Rust acceleration for modularsnf.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ring::RustRingZModN>()?;
    m.add_function(wrap_pyfunction!(diagonal::rust_smith_from_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(diagonal::rust_merge_smith_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(rust_smith_normal_form, m)?)?;
    Ok(())
}

/// Full Smith Normal Form: takes an n×n matrix and modulus,
/// returns (U, V, S) as numpy arrays with S = U @ A @ V (mod N).
#[pyfunction]
fn rust_smith_normal_form<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i64>>, Bound<'py, PyArray2<i64>>)> {
    let r = ring::RustRingZModN::new_internal(modulus)?;
    let a = data.as_array().to_owned();
    let n = a.nrows();
    let m = a.ncols();

    if n == 0 || m == 0 {
        let u = numpy::ndarray::Array2::<i64>::eye(n);
        let v = numpy::ndarray::Array2::<i64>::eye(m);
        return Ok((
            u.into_pyarray(py),
            v.into_pyarray(py),
            a.into_pyarray(py),
        ));
    }

    // Pad to square if needed
    let s = n.max(m);
    let mut a_pad = numpy::ndarray::Array2::zeros((s, s));
    a_pad.slice_mut(numpy::ndarray::s![..n, ..m]).assign(&a);

    let (u_pad, v_pad, s_pad) = snf::smith_square(&a_pad, &r);

    // Crop back
    let u = u_pad.slice(numpy::ndarray::s![..n, ..n]).to_owned();
    let v = v_pad.slice(numpy::ndarray::s![..m, ..m]).to_owned();
    let s_mat = s_pad.slice(numpy::ndarray::s![..n, ..m]).to_owned();

    Ok((
        u.into_pyarray(py),
        v.into_pyarray(py),
        s_mat.into_pyarray(py),
    ))
}
