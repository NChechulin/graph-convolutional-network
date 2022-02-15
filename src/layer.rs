extern crate nalgebra as na;
use na::{Const, DMatrix, Dynamic, Matrix, RowDVector, SliceStorage};

#[derive(Debug)]
pub struct ConvolutionLayer<'a> {
    nodes_num: usize,
    input_features_num: usize,
    output_features_num: usize,
    laplacian: &'a DMatrix<f32>,
    weights: DMatrix<f32>,
}

impl<'a> ConvolutionLayer<'a> {
    /// Creates a new layer with random initialized weights
    /// Arguments:
    /// * `nodes_num` - The number of nodes in the graph
    /// * `input_features_num` - How many features does each node has.
    ///     Should be same as `output_features_num` of the previous layer
    ///     or be equal to the number of features in the dataset if it's the 1st layer
    /// * `output_features_num` - The number of features in the output feature matrix
    pub fn new(
        nodes_num: usize,
        input_features_num: usize,
        output_features_num: usize,
        laplacian: &'a DMatrix<f32>,
    ) -> Self {
        ConvolutionLayer {
            nodes_num,
            input_features_num,
            output_features_num,
            laplacian,
            weights: utils::generate_glorot_matrix(input_features_num, output_features_num),
        }
    }

    /// Applies weights to a given feature matrix of size `nodes_num x input_features_num` and returns a matrix of size `nodes_num x output_features_num`
    pub fn apply(self, feature_matrix: &DMatrix<f32>) -> DMatrix<f32> {
        self.laplacian * feature_matrix * self.weights
    }
}

pub struct SoftmaxLayer {}

impl SoftmaxLayer {
    pub fn softmax(
        row: &Matrix<f32, Const<1_usize>, Dynamic, SliceStorage<'_, f32, Const<1_usize>, Dynamic, Const<1_usize>, Dynamic>>,
    ) -> RowDVector<f32> {
        // FIXME
        // OK at this point I am basically out of my mind
        // and cannot pass a proper RowDVector into this function
        // so I decided to take a shortcut and pass a 1xK matrix
        
        // assert!(row.len() == 1, "Row len: {}", row.len());
        let mut sum: f32 = 0.0;

        for &el in row {
            sum += el.exp();
        }

        RowDVector::from_fn(row.len(), |_, el| row[el].exp() / sum)
    }

    pub fn apply(feature_matrix: &DMatrix<f32>) -> DMatrix<f32> {
        let mut rows = vec![];

        for i in 0..feature_matrix.nrows() {
            rows.push(Self::softmax(&feature_matrix.row(i)));
        }

        DMatrix::<f32>::from_rows(&rows)
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

    pub fn build_laplacian(adj_matrix: &DMatrix<f32>) -> DMatrix<f32> {
        let n = adj_matrix.len();
        std::assert!(n > 0, "Graph should have at least one node!");

        let mut res = DMatrix::<f32>::zeros(n, n);

        for i in 0..n {
            // FIXME: fix i-th row sum
            let mut row_sum: f32 = 0.0;
            for j in 0..n {
                row_sum += adj_matrix[(i, j)];
            }
            res[(i, i)] = row_sum;
        }

        res
    }
}
