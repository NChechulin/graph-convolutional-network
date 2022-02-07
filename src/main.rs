mod layer;
use layer::utils::generate_glorot_matrix;
fn main() {
    let matrix = generate_glorot_matrix(5, 8);
    println!("{}", matrix);
}
