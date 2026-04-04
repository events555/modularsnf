pub mod band;
pub mod diagonal;
pub mod echelon;
pub mod ring;
pub mod snf;

pub use ring::RingZModN;

use ndarray::Array2;

/// Full Smith Normal Form: takes an n x m matrix and modulus,
/// returns (U, V, S) with S = U @ A @ V (mod N).
pub fn smith_normal_form(
    a: &Array2<i64>,
    modulus: i64,
) -> Result<(Array2<i64>, Array2<i64>, Array2<i64>), String> {
    let r = RingZModN::new(modulus)?;
    let n = a.nrows();
    let m = a.ncols();

    if n == 0 || m == 0 {
        let u = Array2::<i64>::eye(n);
        let v = Array2::<i64>::eye(m);
        return Ok((u, v, a.clone()));
    }

    // Pad to square if needed
    let s = n.max(m);
    let mut a_pad = Array2::zeros((s, s));
    a_pad.slice_mut(ndarray::s![..n, ..m]).assign(a);

    let (u_pad, v_pad, s_pad) = snf::smith_square(&a_pad, &r);

    // Crop back
    let u = u_pad.slice(ndarray::s![..n, ..n]).to_owned();
    let v = v_pad.slice(ndarray::s![..m, ..m]).to_owned();
    let s_mat = s_pad.slice(ndarray::s![..n, ..m]).to_owned();

    Ok((u, v, s_mat))
}
