//! ═══════════════════════════════════════════════════════════════════════════════
//! AEGIS Autograd Engine (Ag) - The Bridge
//! ═══════════════════════════════════════════════════════════════════════════════
//!
//! "History is not a tree, it is a tape."
//!
//! A Wengert List (Tape-based) implementation of reverse-mode automatic 
//! differentiation. It records operations on `Gc<Tensor>` handles, allowing
//! the graph to be stored efficiently in the Manifold Heap.
//!
//! ═══════════════════════════════════════════════════════════════════════════════

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::vec::Vec;
#[cfg(feature = "std")]
use std::vec::Vec;

use crate::memory::{Gc, ManifoldHeap};
use crate::ml::tensor::Tensor;

/// Fundamental differentiable operations.
/// Stored as a linear sequence in the GradTape.
#[derive(Debug, Clone, Copy)]
pub enum Op {
    /// out = lhs + rhs
    Add { out: Gc<Tensor>, lhs: Gc<Tensor>, rhs: Gc<Tensor> },
    
    /// out = lhs * rhs (element-wise)
    Mul { out: Gc<Tensor>, lhs: Gc<Tensor>, rhs: Gc<Tensor> },
    
    /// out = lhs @ rhs (matrix multiplication)
    MatMul { out: Gc<Tensor>, lhs: Gc<Tensor>, rhs: Gc<Tensor> },
    
    /// out = max(0, input)
    ReLU { out: Gc<Tensor>, input: Gc<Tensor> },
}

/// The Gradient Tape.
/// Records the history of operations for the backward pass.
pub struct GradTape {
    /// Linear sequence of operations
    pub ops: Vec<Op>,
}

impl GradTape {
    pub fn new() -> Self {
        Self {
            ops: Vec::with_capacity(1024),
        }
    }

    /// Record an operation
    pub fn push(&mut self, op: Op) {
        self.ops.push(op);
    }

    /// Clear the tape (usually after backward)
    pub fn clear(&mut self) {
        self.ops.clear();
    }
}

impl Default for GradTape {
    fn default() -> Self {
        Self::new()
    }
}

/// A Differentiable Variable.
/// Lightweight wrapper around a Heap Handle (Gc).
/// Gradients are computed externally during the backward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Variable {
    pub data: Gc<Tensor>,
}

impl Variable {
    /// Create a new variable handle
    pub fn new(data: Gc<Tensor>) -> Self {
        Self { data }
    }
}

/// The Autograd Context.
/// Manages the interaction between the Heap (data) and the Tape (history).
pub struct Context<'a> {
    pub heap: &'a mut ManifoldHeap<Tensor>,
    pub tape: &'a mut GradTape,
}

impl<'a> Context<'a> {
    pub fn new(heap: &'a mut ManifoldHeap<Tensor>, tape: &'a mut GradTape) -> Self {
        Self { heap, tape }
    }

    /// Allocate a new variable (leaf)
    pub fn var(&mut self, tensor: Tensor) -> Variable {
        Variable::new(self.heap.alloc(tensor))
    }

    /// Add two variables
    pub fn add(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let a = self.heap.get(lhs.data).expect("Stale handle LHS");
        let b = self.heap.get(rhs.data).expect("Stale handle RHS");
        let result = a.add(b);
        let out = self.heap.alloc(result);
        
        self.tape.push(Op::Add { 
            out, 
            lhs: lhs.data, 
            rhs: rhs.data 
        });
        
        Variable::new(out)
    }

    /// Multiply two variables
    pub fn mul(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let a = self.heap.get(lhs.data).expect("Stale handle LHS");
        let b = self.heap.get(rhs.data).expect("Stale handle RHS");
        let result = a.mul(b);
        let out = self.heap.alloc(result);
        
        self.tape.push(Op::Mul { 
            out, 
            lhs: lhs.data, 
            rhs: rhs.data 
        });
        
        Variable::new(out)
    }

    /// Matrix Multiplication
    pub fn matmul(&mut self, lhs: Variable, rhs: Variable) -> Variable {
        let a = self.heap.get(lhs.data).expect("Stale handle LHS");
        let b = self.heap.get(rhs.data).expect("Stale handle RHS");
        let result = a.matmul(b);
        let out = self.heap.alloc(result);
        
        self.tape.push(Op::MatMul { 
            out, 
            lhs: lhs.data, 
            rhs: rhs.data 
        });
        
        Variable::new(out)
    }

    /// ReLU Activation
    pub fn relu(&mut self, input: Variable) -> Variable {
        let val = self.heap.get(input.data).expect("Stale handle");
        let result = val.map(|x| if x > 0.0 { x } else { 0.0 });
        let out = self.heap.alloc(result);
        
        self.tape.push(Op::ReLU { 
            out, 
            input: input.data 
        });
        
        Variable::new(out)
    }

    /// Reverse-Mode Backward Pass
    /// Returns the gradients for all used variables as a Map (Vec indexed by Gc.index)
    pub fn backward(&mut self, target: Variable) -> Vec<Option<Tensor>> {
        let max_idx = self.heap.capacity();
        #[cfg(not(feature = "std"))]
        let mut grads: Vec<Option<Tensor>> = alloc::vec![None; max_idx];
        #[cfg(feature = "std")]
        let mut grads: Vec<Option<Tensor>> = std::vec![None; max_idx];
        
        // Seed output gradient
        let target_tensor = self.heap.get(target.data).expect("Target lost");
        grads[target.data.index] = Some(Tensor::ones(&target_tensor.shape));

        // Iterate tape in reverse
        for op in self.tape.ops.iter().rev() {
            match op {
                Op::Add { out, lhs, rhs } => {
                     // Solves borrow checker by cloning Option first
                     let grad_out = grads[out.index].clone();
                     if let Some(grad) = grad_out {
                         // dL/d(lhs) += dL/dout * 1
                         Self::accumulate_grad(&mut grads, lhs.index, &grad);
                         Self::accumulate_grad(&mut grads, rhs.index, &grad);
                     }
                },
                Op::Mul { out, lhs, rhs } => {
                    let grad_out = grads[out.index].clone();
                    if let Some(grad) = grad_out {
                        let lhs_val = self.heap.get(*lhs).unwrap();
                        let rhs_val = self.heap.get(*rhs).unwrap();
                        
                        // dL/dLhs = grad_out * rhs
                        let d_lhs: Tensor = grad.mul(rhs_val);
                        Self::accumulate_grad(&mut grads, lhs.index, &d_lhs);
                        
                        // dL/dRhs = grad_out * lhs
                        let d_rhs: Tensor = grad.mul(lhs_val);
                        Self::accumulate_grad(&mut grads, rhs.index, &d_rhs);
                    }
                },
                Op::MatMul { out, lhs, rhs } => {
                     let grad_out = grads[out.index].clone();
                     if let Some(grad) = grad_out {
                        let lhs_val = self.heap.get(*lhs).unwrap();
                        let rhs_val = self.heap.get(*rhs).unwrap();
                        
                        // C = A @ B
                        // dA = dC @ B^T
                        let d_lhs: Tensor = grad.matmul(&rhs_val.transpose());
                        Self::accumulate_grad(&mut grads, lhs.index, &d_lhs);
                        
                        // dB = A^T @ dC
                        let d_rhs: Tensor = lhs_val.transpose().matmul(&grad);
                        Self::accumulate_grad(&mut grads, rhs.index, &d_rhs);
                     }
                },
                Op::ReLU { out, input } => {
                    let grad_out = grads[out.index].clone();
                    if let Some(grad) = grad_out {
                        let input_val = self.heap.get(*input).unwrap();
                        // dL/dx = grad_out * (1 if x > 0 else 0)
                        let mask = input_val.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
                        let d_input: Tensor = grad.mul(&mask);
                        Self::accumulate_grad(&mut grads, input.index, &d_input);
                    }
                }
            }
        }
        
        grads
    }
    
    fn accumulate_grad(grads: &mut Vec<Option<Tensor>>, idx: usize, grad: &Tensor) {
        if idx >= grads.len() {
            grads.resize(idx + 1 + 256, None);
        }
        
        match &mut grads[idx] {
            Some(existing) => {
                let new = existing.add(grad);
                grads[idx] = Some(new);
            }
            None => {
                grads[idx] = Some(grad.clone());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::ManifoldHeap;
    use crate::ml::tensor::Tensor;

    #[test]
    fn test_autograd_simple_mul() {
        let mut heap = ManifoldHeap::<Tensor>::new();
        let mut tape = GradTape::new();
        let mut ctx = Context::new(&mut heap, &mut tape);

        let a = ctx.var(Tensor::new(&[2.0], &[1]));
        let b = ctx.var(Tensor::new(&[3.0], &[1]));
        let c = ctx.mul(a, b);

        let grads = ctx.backward(c);

        let da = grads[a.data.index].as_ref().unwrap();
        let db = grads[b.data.index].as_ref().unwrap();

        assert_eq!(da.get(&[0]), 3.0);
        assert_eq!(db.get(&[0]), 2.0);
    }

    #[test]
    fn test_autograd_complex() {
        // y = x^2 + x, at x=2. dy/dx = 2x + 1 = 5.
        let mut heap = ManifoldHeap::<Tensor>::new();
        let mut tape = GradTape::new();
        let mut ctx = Context::new(&mut heap, &mut tape);

        let x = ctx.var(Tensor::new(&[2.0], &[1]));
        let x2 = ctx.mul(x, x);
        let y = ctx.add(x2, x);

        let grads = ctx.backward(y);
        let dx = grads[x.data.index].as_ref().unwrap();

        assert_eq!(dx.get(&[0]), 5.0);
    }

    #[test]
    fn test_xor_backprop() {
        let mut heap = ManifoldHeap::<Tensor>::new();
        let mut tape = GradTape::new();
        let mut ctx = Context::new(&mut heap, &mut tape);

        // Input: [1.0, 1.0] (1x2)
        let x = ctx.var(Tensor::new(&[1.0, 1.0], &[1, 2]));
        
        // W1: Identity [[1, 0], [0, 1]] (2x2)
        let w1 = ctx.var(Tensor::new(&[1.0, 0.0, 0.0, 1.0], &[2, 2]));
        
        // W2: [[1], [-2]] (2x1)
        let w2 = ctx.var(Tensor::new(&[1.0, -2.0], &[2, 1]));

        // Forward
        // z1 = x @ W1 = [1, 1]
        let z1 = ctx.matmul(x, w1);
        
        // h1 = relu(z1) = [1, 1]
        let h1 = ctx.relu(z1);
        
        // y = h1 @ W2 = 1*1 + 1*-2 = -1 (1x1)
        let y = ctx.matmul(h1, w2);
        
        // Loss = (y - target)^2, target=0
        // Loss = y * y
        let loss = ctx.mul(y, y);
        
        // Backward
        // dLoss/dy = 2y = -2
        let grads = ctx.backward(loss);
        
        // Checks
        let dy = grads[y.data.index].as_ref().unwrap();
        // Manually: dL/dy = 2*(-1) = -2
        assert!((dy.get(&[0,0]) - (-2.0)).abs() < 1e-6);
        
        let dw2 = grads[w2.data.index].as_ref().unwrap();
        // dL/dW2 = h1.T @ dy = [[1], [1]] @ [-2] = [[-2], [-2]]
        assert!((dw2.get(&[0,0]) - (-2.0)).abs() < 1e-6);
        assert!((dw2.get(&[1,0]) - (-2.0)).abs() < 1e-6);
    }
}
