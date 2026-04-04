use modularsnf::ring::RingZModN;
use modularsnf::snf::smith_square;
use modularsnf::diagonal::{merge_raw, smith_from_diagonal};

use numpy::ndarray::{s, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Wrapper around RingZModN for Python.
#[pyclass]
#[pyo3(name = "RustRingZModN")]
struct PyRingZModN {
    inner: RingZModN,
}

#[pymethods]
impl PyRingZModN {
    #[new]
    fn new(n: i64) -> PyResult<Self> {
        let inner = RingZModN::new(n).map_err(PyValueError::new_err)?;
        Ok(Self { inner })
    }

    #[getter]
    fn n_val(&self) -> i64 {
        self.inner.n()
    }

    fn add(&self, a: i64, b: i64) -> i64 {
        self.inner.add(a, b)
    }

    fn sub(&self, a: i64, b: i64) -> i64 {
        self.inner.sub(a, b)
    }

    fn mul(&self, a: i64, b: i64) -> i64 {
        self.inner.mul(a, b)
    }

    fn is_zero(&self, a: i64) -> bool {
        self.inner.is_zero(a)
    }

    fn gcd(&self, a: i64, b: i64) -> i64 {
        self.inner.gcd(a, b)
    }

    fn ass(&self, a: i64) -> i64 {
        self.inner.ass(a)
    }

    fn ann(&self, a: i64) -> i64 {
        self.inner.ann(a)
    }

    fn rem(&self, a: i64, b: i64) -> i64 {
        self.inner.rem(a, b)
    }

    fn quo(&self, a: i64, b: i64) -> PyResult<i64> {
        self.inner.quo(a, b).map_err(PyValueError::new_err)
    }

    fn div(&self, a: i64, b: i64) -> PyResult<i64> {
        self.inner.div(a, b).map_err(PyValueError::new_err)
    }

    fn unit(&self, a: i64) -> i64 {
        self.inner.unit(a)
    }

    fn gcdex(&self, a: i64, b: i64) -> PyResult<(i64, i64, i64, i64, i64)> {
        Ok(self.inner.gcdex(a, b))
    }

    fn stab(&self, a: i64, b: i64, c: i64) -> PyResult<i64> {
        self.inner.stab(a, b, c).map_err(PyValueError::new_err)
    }
}

/// Full Smith Normal Form: takes an n x m matrix and modulus,
/// returns (U, V, S) as numpy arrays with S = U @ A @ V (mod N).
#[pyfunction]
fn rust_smith_normal_form<'py>(
    py: Python<'py>,
    data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let r = RingZModN::new(modulus).map_err(PyValueError::new_err)?;
    let a = data.as_array().to_owned();
    let n = a.nrows();
    let m = a.ncols();

    if n == 0 || m == 0 {
        let u = Array2::<i64>::eye(n);
        let v = Array2::<i64>::eye(m);
        return Ok((u.into_pyarray(py), v.into_pyarray(py), a.into_pyarray(py)));
    }

    // Pad to square if needed
    let sz = n.max(m);
    let mut a_pad = Array2::zeros((sz, sz));
    a_pad.slice_mut(s![..n, ..m]).assign(&a);

    let (u_pad, v_pad, s_pad) = smith_square(&a_pad, &r);

    // Crop back
    let u = u_pad.slice(s![..n, ..n]).to_owned();
    let v = v_pad.slice(s![..m, ..m]).to_owned();
    let s_mat = s_pad.slice(s![..n, ..m]).to_owned();

    Ok((
        u.into_pyarray(py),
        v.into_pyarray(py),
        s_mat.into_pyarray(py),
    ))
}

/// Compute SNF of a diagonal matrix.
#[pyfunction]
fn rust_smith_from_diagonal<'py>(
    py: Python<'py>,
    diag_data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let ring = RingZModN::new(modulus).map_err(PyValueError::new_err)?;
    let arr = diag_data.as_array().to_owned();
    let (u, v, s) = smith_from_diagonal(&arr, &ring);

    Ok((
        u.into_pyarray(py),
        v.into_pyarray(py),
        s.into_pyarray(py),
    ))
}

/// Merge two SNF blocks.
#[pyfunction]
fn rust_merge_smith_blocks<'py>(
    py: Python<'py>,
    a_data: PyReadonlyArray2<'py, i64>,
    b_data: PyReadonlyArray2<'py, i64>,
    modulus: i64,
) -> PyResult<(
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
)> {
    let ring = RingZModN::new(modulus).map_err(PyValueError::new_err)?;
    let a = a_data.as_array().to_owned();
    let b = b_data.as_array().to_owned();

    let (u, v, s) = merge_raw(&a, &b, &ring);

    Ok((u.into_pyarray(py), v.into_pyarray(py), s.into_pyarray(py)))
}

/// Native Rust acceleration for modularsnf.
#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRingZModN>()?;
    m.add_function(wrap_pyfunction!(rust_smith_from_diagonal, m)?)?;
    m.add_function(wrap_pyfunction!(rust_merge_smith_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(rust_smith_normal_form, m)?)?;
    Ok(())
}
