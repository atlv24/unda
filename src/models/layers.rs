use crate::graph::{constant, dtypes, Context, Node, NodeIdentifier, Operation, Result, Shape};

use super::initializers::{ConstInit, Initializer};

pub struct ConvParams<T> {
    kernel: T,
    bias: T,
}

pub struct BatchNormParams<T> {
    mu: T,
    sigma: T,
    alpha: T,
    beta: T,
}

impl Context {
    pub fn dense<IW: Initializer, IB: Initializer>(
        &mut self,
        input_node: NodeIdentifier,
        out_size: u32,
        kernel_initializer: IW,
        bias_initializer: IB,
        kernel_seed: i64,
        bias_seed: i64,
        name: &str,
    ) -> Result<(
        NodeIdentifier,
        ConvParams<NodeIdentifier>,
        ConvParams<xla::Literal>,
    )> {
        let shape = self.nodes[input_node].shape.clone();
        let last_dim = shape.sizes[shape.ndims() - 1];
        let dtype = self.nodes[input_node].dtype;

        let kernel_shape = Shape::from([last_dim, out_size]);
        let mut kernel_name = name.to_owned();
        kernel_name.push_str("_kernel");
        let kernel_val = kernel_initializer.initialize(kernel_seed, &kernel_shape, dtype)?;
        let kernel_id = self.parameter(kernel_name, kernel_shape, dtype)?;

        let mut bias_shape = Shape::new();
        for _ in 0..(shape.ndims() - 1) {
            bias_shape.sizes.push(1u32);
        }
        bias_shape.sizes.push(out_size);
        let mut bias_name = name.to_owned();
        bias_name.push_str("_bias");
        let bias_val = bias_initializer.initialize(bias_seed, &bias_shape, dtype)?;
        let bias_id = self.parameter(bias_name, bias_shape, dtype)?;

        let matmul_node = self.matmul(input_node, kernel_id)?;
        let dense_node = self.add(matmul_node, bias_id)?;

        Ok((
            dense_node,
            ConvParams {
                kernel: kernel_id,
                bias: bias_id,
            },
            ConvParams {
                kernel: kernel_val,
                bias: bias_val,
            },
        ))
    }

    pub fn batchnorm(
        &mut self,
        input_node: NodeIdentifier,
        eps: f32,
        name: &str,
    ) -> Result<(
        NodeIdentifier,
        BatchNormParams<NodeIdentifier>,
        BatchNormParams<xla::Literal>,
    )> {
        let shape = self.nodes[input_node].shape.clone();
        let last_dim = shape.sizes[shape.ndims() - 1];
        let dtype = self.nodes[input_node].dtype;

        check_fp_type(dtype)?;

        let mut param_shapes = shape.clone();
        for i in 0..param_shapes.ndims() - 1 {
            param_shapes.sizes[i] = 1;
        }

        let mut mu_name = name;
        mu_name.push_str("_mu");
        let mu = self.parameter(mu_name, param_shapes, dtype)?;

        let mut sigma_name = name;
        sigma_name.push_str("_sigma");
        let sigma = self.parameter(sigma_name, param_shapes, dtype)?;

        let mut alpha_name = name;
        alpha_name.push_str("_alpha");
        let alpha = self.parameter(alpha_name, param_shapes, dtype)?;

        let mut beta_name = name;
        beta_name.push_str("_beta");
        let beta = self.parameter(beta_name, param_shapes, dtype)?;

        let epsilon = self.scalar(epsilon, dtype)?;

        let mu_literal = ConstInit(0.0).initialize(0, param_shapes, dtype)?;
        let beta_literal = ConstInit(0.0).initialize(0, param_shapes, dtype)?;
        let alpha_literal = ConstInit(1.0).initialize(0, param_shapes, dtype)?;
        let sigma_literal = ConstInit(1.0).initialize(0, param_shapes, dtype)?;

        let batchnorm_node = self.nodes.insert(Operation::BatchNorm {
            mu,
            sigma,
            epsilon,
            alpha,
            beta,
            x: input_node,
        });
    }
}
