#[derive(Clone, Debug)]
pub struct LinearLayer {
    pub weight: Vec<Vec<f32>>, // shape [out_dim][in_dim]
    pub bias: Option<Vec<f32>>, // len out_dim
}

impl LinearLayer {
    pub fn new(weight: Vec<Vec<f32>>, bias: Option<Vec<f32>>) -> Self {
        Self { weight, bias }
    }

    pub fn forward(&self, input: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        let out_dim = self.weight.len();
        let in_dim = if out_dim > 0 { self.weight[0].len() } else { 0 };
        let mut output = Vec::with_capacity(input.len());
        for row in input.iter() {
            let mut out_row = vec![0f32; out_dim];
            for j in 0..out_dim {
                let mut sum = match &self.bias {
                    Some(b) => b[j],
                    None => 0.0,
                };
                for k in 0..in_dim {
                    sum += row[k] * self.weight[j][k];
                }
                out_row[j] = sum;
            }
            output.push(out_row);
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct SiluAndMul;

impl SiluAndMul {
    pub fn new() -> Self {
        Self
    }

    pub fn forward(&self, input: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if input.is_empty() { return Vec::new(); }
        let dim = input[0].len() / 2;
        let mut output = Vec::with_capacity(input.len());
        for row in input.iter() {
            let mut out_row = vec![0f32; dim];
            for j in 0..dim {
                let x = row[j];
                let y = row[j + dim];
                let silu = x / (1.0 + (-x).exp());
                out_row[j] = silu * y;
            }
            output.push(out_row);
        }
        output
    }
}

#[derive(Clone, Debug)]
pub struct RMSNorm {
    pub weight: Vec<f32>,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self { weight: vec![1.0; hidden_size], eps }
    }

    pub fn forward(&self, input: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
        if input.is_empty() { return Vec::new(); }
        let hs = self.weight.len() as f32;
        let mut output = Vec::with_capacity(input.len());
        for row in input.iter() {
            let var: f32 = row.iter().map(|v| v * v).sum::<f32>() / hs;
            let inv_rms = (var + self.eps).sqrt().recip();
            let mut out_row = Vec::with_capacity(self.weight.len());
            for (j, w) in self.weight.iter().enumerate() {
                out_row.push(row[j] * inv_rms * *w);
            }
            output.push(out_row);
        }
        output
    }
}
