//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Convolutional Layers
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! 2D Convolution implementation for image processing and spatial pattern recognition.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#![allow(dead_code)]

use libm::sqrt;
use crate::ml::neural::Activation;

/// Maximum kernel size (e.g., 3x3, 5x5)
const MAX_KERNEL_SIZE: usize = 5;
/// Maximum input channels
const MAX_CHANNELS_IN: usize = 3;
/// Maximum output channels (filters)
const MAX_CHANNELS_OUT: usize = 8;
/// Maximum image dimension
const MAX_IMG_DIM: usize = 32;

/// 2D Convolutional Layer
#[derive(Debug, Clone)]
pub struct Conv2D {
    /// Filters [out_channel][in_channel][k_y][k_x]
    pub weights: [[[[f64; MAX_KERNEL_SIZE]; MAX_KERNEL_SIZE]; MAX_CHANNELS_IN]; MAX_CHANNELS_OUT],
    /// Biases [out_channel]
    pub biases: [f64; MAX_CHANNELS_OUT],
    /// Input channels
    pub in_channels: usize,
    /// Output channels
    pub out_channels: usize,
    /// Kernel size (k x k)
    pub kernel_size: usize,
    /// Stride
    pub stride: usize,
    /// Padding
    pub padding: usize,
    /// Activation
    pub activation: Activation,
    
    // Cache for backprop
    pub last_input: [[[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN]; 1], // Batch size 1 for now
    pub last_output_dim: (usize, usize),
}

impl Conv2D {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: Activation,
    ) -> Self {
        let in_c = in_channels.min(MAX_CHANNELS_IN);
        let out_c = out_channels.min(MAX_CHANNELS_OUT);
        let k_size = kernel_size.min(MAX_KERNEL_SIZE);

        // Kaiming/He initialization
        let scale = sqrt(2.0 / (in_c * k_size * k_size) as f64);
        
        let mut weights = [[[[0.0; MAX_KERNEL_SIZE]; MAX_KERNEL_SIZE]; MAX_CHANNELS_IN]; MAX_CHANNELS_OUT];
        let mut rng = 42u64;

        for channel_out in weights.iter_mut().take(out_c) {
            for channel_in in channel_out.iter_mut().take(in_c) {
                for row in channel_in.iter_mut().take(k_size) {
                    for val in row.iter_mut().take(k_size) {
                        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                        let r = (rng as f64 / u64::MAX as f64) * 2.0 - 1.0;
                        *val = r * scale;
                    }
                }
            }
        }

        Self {
            weights,
            biases: [0.0; MAX_CHANNELS_OUT],
            in_channels: in_c,
            out_channels: out_c,
            kernel_size: k_size,
            stride,
            padding,
            activation,
            last_input: [[[[0.0; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN]; 1],
            last_output_dim: (0, 0),
        }
    }

    /// Forward pass
    /// input: [channels][height][width]
    pub fn forward(
        &mut self, 
        input: &[[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN],
        input_h: usize,
        input_w: usize
    ) -> ([[[f64; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_OUT], usize, usize) {
        // Cache input
        self.last_input[0] = *input;

        let output_h = (input_h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let output_w = (input_w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        self.last_output_dim = (output_h, output_w);

        let mut output = [[[0.0; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_OUT];

        for (o, channel_out) in output.iter_mut().enumerate().take(self.out_channels) {
            for (y, row_out) in channel_out.iter_mut().enumerate().take(output_h) {
                for (x, val_out) in row_out.iter_mut().enumerate().take(output_w) {
                    let mut sum = self.biases[o];
                    
                    // Convolve
                    let in_y_origin = (y * self.stride) as isize - self.padding as isize;
                    let in_x_origin = (x * self.stride) as isize - self.padding as isize;

                    for (c, input_channel) in input.iter().enumerate().take(self.in_channels) {
                        for ky in 0..self.kernel_size {
                            for kx in 0..self.kernel_size {
                                let in_y = in_y_origin + ky as isize;
                                let in_x = in_x_origin + kx as isize;

                                if in_y >= 0 && in_y < input_h as isize && 
                                   in_x >= 0 && in_x < input_w as isize {
                                    sum += input_channel[in_y as usize][in_x as usize] * 
                                           self.weights[o][c][ky][kx];
                                }
                            }
                        }
                    }

                    // Activation
                    *val_out = self.activation.apply_scalar(sum);
                }
            }
        }

        (output, output_h, output_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_initialization() {
        let conv = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU);
        assert_eq!(conv.weights.len(), MAX_CHANNELS_OUT); // Array size fixed
        // Check params
        assert_eq!(conv.kernel_size, 3);
        assert_eq!(conv.stride, 1);
        assert_eq!(conv.padding, 1);
    }

    #[test]
    fn test_conv2d_forward_shape() {
        let mut conv = Conv2D::new(1, 1, 3, 1, 1, Activation::ReLU);
        let input = [[[0.5; MAX_IMG_DIM]; MAX_IMG_DIM]; MAX_CHANNELS_IN];
        
        // 10x10 input
        // Output size: (10 + 2*1 - 3) / 1 + 1 = 10
        let (_, h, w) = conv.forward(&input, 10, 10);
        
        assert_eq!(h, 10);
        assert_eq!(w, 10);
    }
}
