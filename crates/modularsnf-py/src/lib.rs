use modularsnf::crt::crt_snf;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

type SnfArrays<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
);

/// Smith Normal Form via Storjohann band reduction. Takes an n x m matrix and
/// modulus; returns (U, V, S) as numpy arrays with S = U @ A @ V (mod N).
#[pyfunction]
fn rust_smith_normal_form<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<SnfArrays<'py>> {
    let a = data.as_array().to_owned();
    let (u, v, s) = modularsnf::smith_normal_form(&a, modulus).map_err(PyValueError::new_err)?;

    Ok((u.into_pyarray(py), v.into_pyarray(py), s.into_pyarray(py)))
}

/// CRT fast-path SNF. Takes an n x m matrix, modulus, and the prime
/// factorization of the modulus as (p, e) pairs (so factoring is amortized).
/// Returns (U, V, S) with S = U @ A @ V (mod N).
#[pyfunction]
fn rust_crt_snf<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
    factors: Vec<(i64, u32)>,
) -> PyResult<SnfArrays<'py>> {
    let a = data.as_array().to_owned();

    if a.nrows() == 0 || a.ncols() == 0 {
        let u = Array2::<i64>::eye(a.nrows());
        let v = Array2::<i64>::eye(a.ncols());
        return Ok((u.into_pyarray(py), v.into_pyarray(py), a.into_pyarray(py)));
    }

    let (u, v, s) = crt_snf(&a, modulus, &factors);

    Ok((u.into_pyarray(py), v.into_pyarray(py), s.into_pyarray(py)))
}

/// Native Rust acceleration for modularsnf.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rust_smith_normal_form, m)?)?;
    m.add_function(wrap_pyfunction!(rust_crt_snf, m)?)?;
    Ok(())
}
