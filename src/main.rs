mod layer;
use layer::ConvolutionLayer;
fn main() {
    let layer = ConvolutionLayer::new(2000, 1403, 16);
    println!("{:?}", layer);
}
