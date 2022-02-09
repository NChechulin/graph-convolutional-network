mod layer;
use layer::ConvolutionLayer;
extern crate nalgebra as na;
use na::{DMatrix, Matrix5x4};

fn sample_laplacian() -> DMatrix<f32> {
    DMatrix::from_vec(
        5,
        5,
        vec![
            0.25, 0.25, 0.29, 0., 0.29, 0.25, 0.25, 0., 0.29, 0.29, 0.29, 0., 0.33, 0., 0.33, 0.,
            0.29, 0., 0.33, 0., 0.29, 0.29, 0.33, 0., 0.33,
        ],
    )
}

fn sample_feature_matrix() -> DMatrix<f32> {
    DMatrix::from_vec(
        5,
        4,
        vec![
            4.2, 2.83, 4.44, 2.96, 0.7, 3.98, 0.63, 0.63, 2.38, 3.85, 3.29, 2.23, 3.43, 2.4, 0.14,
            0.59, 1.83, 0.94, 4.18, 4.45,
        ],
    )
}
fn main() {
    let feature_matrix = sample_feature_matrix();
    let laplacian = sample_laplacian();
    let layer = ConvolutionLayer::new(5, 4, 2, &laplacian);
    let result = layer.apply(&(feature_matrix as DMatrix<f32>));
    println!("{:?}", result);
}
