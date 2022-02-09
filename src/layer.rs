extern crate nalgebra as na;
use na::{DMatrix, DVector};

#[derive(Debug)]
pub struct ConvolutionLayer {
    nodes_num: usize,
    input_features_num: usize,
    output_features_num: usize,
    weights: DMatrix<f32>,
}

impl ConvolutionLayer {
    /// Creates a new layer with random initialized weights
    /// Arguments:
    /// * `nodes_num` - The number of nodes in the graph
    /// * `input_features_num` - How many features does each node has.
    ///     Should be same as `output_features_num` of the previous layer
    ///     or be equal to the number of features in the dataset if it's the 1st layer
    /// * `output_features_num` - The number of features in the output feature matrix
    pub fn new(nodes_num: usize, input_features_num: usize, output_features_num: usize) -> Self {
        ConvolutionLayer {
            nodes_num,
            input_features_num,
            output_features_num,
            weights: utils::generate_glorot_matrix(input_features_num, output_features_num),
        }
    }
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
