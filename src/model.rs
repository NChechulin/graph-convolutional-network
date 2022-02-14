use nalgebra::{DMatrix};

use crate::layer::{ConvolutionLayer};

pub struct GCN<'a> {
    laplacian: &'a DMatrix<f32>,
    layers: Vec<ConvolutionLayer<'a>>,
}

impl<'a> GCN<'a> {
    fn validate_layer_sizes(layer_sizes: &[usize]) {
        std::assert!(
            layer_sizes.len() > 0,
            "At least one layer should be present!"
        );
        for &dim in layer_sizes {
            std::assert!(dim > 0, "Dimension can not be equal to 0!");
        }
    }

    pub fn new<'b>(
        layer_sizes: &[usize],
        adj_matrix: &'a DMatrix<f32>,
        laplacian: &'a DMatrix<f32>,
    ) -> Self {
        Self::validate_layer_sizes(layer_sizes);
        let nodes_num: usize = adj_matrix.len();

        let mut full_layer_sizes: Vec<(usize, usize)> = vec![];
        for i in 0..layer_sizes.len() - 1 {
            full_layer_sizes.push((layer_sizes[i], layer_sizes[i + 1]));
        }

        let mut layers: Vec<ConvolutionLayer> = vec![];
        for (inp, out) in full_layer_sizes {
            layers.push(ConvolutionLayer::new(nodes_num, inp, out, &laplacian));
        }

        GCN { laplacian, layers }
    }
}
