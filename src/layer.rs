extern crate nalgebra as na;
use na::{DMatrix, DVector};

pub struct ConvolutionLayer {
    input_dim: (usize, usize),
    output_dim: (usize, usize),
    weights: DMatrix<f32>,
}

pub mod utils {
    extern crate nalgebra as na;
    extern crate rand;
    use na::DMatrix;
    use rand::distributions::{Distribution, Uniform};

    /// Generates a matrix using Glorot et. al random initialization
    pub fn generate_glorot_matrix(height: usize, width: usize) -> DMatrix<f32> {
        let n: f32 = (width + height) as f32;
        let boundary: f32 = (6.0 / n).sqrt();
        let distribution = Uniform::new_inclusive(-boundary, boundary);
        let mut rng = rand::thread_rng();

        DMatrix::from_fn(height, width, |_r, _c| distribution.sample(&mut rng))
    }
}
