use ndarray::{Array1, Array2};

#[derive(Debug, Clone)]
pub struct Linear {
    pub weight: Array2<f32>,
    pub bias: Option<Array1<f32>>,
}

impl Linear {
    pub fn new(weight: Array2<f32>, bias: Option<Array1<f32>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: &Array2<f32>) -> Array2<f32> {
        let mut y = input.dot(&self.weight.t());
        if let Some(ref b) = self.bias {
            y = y + b;
        }
        y
    }
}
