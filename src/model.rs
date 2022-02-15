use nalgebra::DMatrix;

use crate::layer::{ConvolutionLayer, SoftmaxLayer};

pub struct GCN<'a> {
    laplacian: &'a DMatrix<f32>,
    layers: Vec<ConvolutionLayer<'a>>,
}

impl<'a> GCN<'a> {
    fn validate_layer_sizes(layer_sizes: &[usize]) {
        std::assert!(
            !layer_sizes.is_empty(),
            "At least one layer should be present!"
        );
        for &dim in layer_sizes {
            std::assert!(dim > 0, "Dimension can not be equal to 0!");
        }
    }

    pub fn new(
        layer_sizes: &[usize],
        adj_matrix: &'a DMatrix<f32>,
        laplacian: &'a DMatrix<f32>,
    ) -> Self {
        Self::validate_layer_sizes(layer_sizes);
        let _nodes_num: usize = adj_matrix.nrows();

        let mut full_layer_sizes: Vec<(usize, usize)> = vec![];
        for i in 0..layer_sizes.len() - 1 {
            full_layer_sizes.push((layer_sizes[i], layer_sizes[i + 1]));
        }

        let mut layers: Vec<ConvolutionLayer> = vec![];
        for (inp, out) in full_layer_sizes {
            layers.push(ConvolutionLayer::new(inp, out, &laplacian));
        }

        GCN { laplacian, layers }
    }

    pub fn apply(&self, feature_matrix: &DMatrix<f32>) -> DMatrix<f32> {
        let mut current_input = (*feature_matrix).clone();

        for conv_layer in self.layers.iter() {
            current_input = conv_layer.apply(&current_input);
        }

        SoftmaxLayer::apply(&current_input)
    }
}
