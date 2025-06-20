use ndarray::{Array2};

#[derive(Debug, Default, Clone)]
pub struct SiluAndMul;

impl SiluAndMul {
    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let (batch, features) = input.dim();
        assert!(features % 2 == 0, "input last dim must be even");
        let half = features / 2;
        let mut out = Array2::<f32>::zeros((batch, half));
        for i in 0..batch {
            for j in 0..half {
                let x = input[(i, j)];
                let y = input[(i, j + half)];
                let silu = x * (1.0 / (1.0 + (-x).exp()));
                out[(i, j)] = silu * y;
            }
        }
        out
    }
}
