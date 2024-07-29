use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};

pub trait Optimizer<U> {
    fn get_step(&self) -> &Context;
    fn n_params(&self) -> usize;
    fn state_size(&self) -> usize;
    fn get_old_params(&self) -> &Vec<NodeIdentifier>;
    fn get_gradients(&self) -> &Vec<NodeIdentifier>;
    fn get_new_params(&self) -> &Vec<NodeIdentifier>;
    fn get_old_state(&self) -> &Vec<NodeIdentifier>;
    fn get_new_state(&self) -> &Vec<NodeIdentifier>;
    fn get_user_params(&self) -> U;
    fn initialize(user_params: U, model_params: Vec<NodeIdentifier>, model: &Context) -> Self;
}

pub struct ChainedOptimizer<U> {
    step: Context,
    n_params: usize,
    state_size: usize,
    old_params: Vec<NodeIdentifier>,
    grads: Vec<NodeIdentifier>,
    new_params: Vec<NodeIdentifier>,
    old_state: Vec<NodeIdentifier>,
    new_state: Vec<NodeIdentifier>,
    pub user_params: U,
}

// WILL FAIL!
// TODO: Need to get node identifiers into the merged context!
//*
pub fn chain<U1, U2, O1: Optimizer<U1>, O2: Optimizer<U2>>(
    opt1: O1,
    opt2: O2,
) -> Result<ChainedOptimizer<(U1, U2)>> {
    // is it necessary to clone here???
    let mut step = opt1.get_step().clone();
    step.merge_graphs(opt2.get_step(), &[])?;

    Ok(ChainedOptimizer {
        step,
        n_params: opt1.n_params() + opt2.n_params(),
        state_size: opt1.state_size() + opt2.state_size(),
        old_params: [opt1.get_old_params().clone(), opt2.get_old_params().clone()].concat(),
        grads: [opt1.get_gradients().clone(), opt2.get_gradients().clone()].concat(),
        new_params: [opt1.get_new_params().clone(), opt2.get_new_params().clone()].concat(),
        old_state: [opt1.get_old_state().clone(), opt2.get_old_state().clone()].concat(),
        new_state: [opt1.get_new_state().clone(), opt2.get_new_state().clone()].concat(),
        user_params: (opt1.get_user_params(), opt2.get_user_params()),
    })
}
//*/
pub struct SGD {
    step: Context,
    n_params: usize,
    old_params: Vec<NodeIdentifier>,
    grads: Vec<NodeIdentifier>,
    new_params: Vec<NodeIdentifier>,
    dummy_state: Vec<NodeIdentifier>,
    pub learning_rate: f32,
}

impl Optimizer<f32> for SGD {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn n_params(&self) -> usize {
        self.n_params
    }
    fn state_size(&self) -> usize {
        0
    }
    fn get_old_params(&self) -> &Vec<NodeIdentifier> {
        &self.old_params
    }
    fn get_gradients(&self) -> &Vec<NodeIdentifier> {
        &self.grads
    }
    fn get_new_params(&self) -> &Vec<NodeIdentifier> {
        &self.new_params
    }
    fn get_old_state(&self) -> &Vec<NodeIdentifier> {
        &self.dummy_state
    }
    fn get_new_state(&self) -> &Vec<NodeIdentifier> {
        &self.dummy_state
    }
    fn get_user_params(&self) -> f32 {
        self.learning_rate
    }
    fn initialize(learning_rate: f32, model_params: Vec<NodeIdentifier>, model: &Context) -> SGD {
        let build = || {
            let mut step = Context::new();

            let n_params = model_params.len();

            let dtype = model.nodes[model_params[0]].dtype;
            let lr = step.scalar(learning_rate, dtype)?;

            let mut old_params = Vec::new();
            let mut grads = Vec::new();
            let mut new_params = Vec::new();

            for (i, node_id) in model_params.iter().enumerate() {
                let model_param = &model.nodes[*node_id];
                let old_param = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let grad = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let mul = step.mul(lr, grad)?;
                let new_param = step.sub(old_param, mul)?;
                old_params.push(old_param);
                grads.push(grad);
                new_params.push(new_param);
            }

            Ok::<SGD, ContextError>(SGD {
                step,
                n_params,
                old_params,
                grads,
                new_params,
                dummy_state: Vec::new(),
                learning_rate,
            })
        };

        build().expect("Failed to initialize SGD")
    }
}

pub struct BatchNormOptimizer {
    step: Context,
    old_params: Vec<NodeIdentifier>,
    grads: Vec<NodeIdentifier>,
    new_params: Vec<NodeIdentifier>,
    dummy_state: Vec<NodeIdentifier>,
    pub momentum: f32,
}

impl Optimizer<f32> for BatchNormOptimizer {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn n_params(&self) -> usize {
        2
    }
    fn state_size(&self) -> usize {
        0
    }
    fn get_old_params(&self) -> &Vec<NodeIdentifier> {
        &self.old_params
    }
    fn get_gradients(&self) -> &Vec<NodeIdentifier> {
        &self.grads
    }
    fn get_new_params(&self) -> &Vec<NodeIdentifier> {
        &self.new_params
    }
    fn get_old_state(&self) -> &Vec<NodeIdentifier> {
        &self.dummy_state
    }
    fn get_new_state(&self) -> &Vec<NodeIdentifier> {
        &self.dummy_state
    }
    fn get_user_params(&self) -> f32 {
        self.momentum
    }
    fn initialize(
        momentum: f32,
        batchnorm_params: Vec<NodeIdentifier>,
        model: &Context,
    ) -> BatchNormOptimizer {
        let build = || {
            let mut step = Context::new();

            let (mu, sigma, x) = (
                batchnorm_params[0],
                batchnorm_params[1],
                batchnorm_params[2],
            );

            let dtype = model.nodes[mu].dtype;
            let mom = step.scalar(momentum, dtype)?;
            let mut mom1 = step.scalar(1.0, dtype)?;
            mom1 = step.sub(mom1, mom)?;

            let old_mu =
                step.parameter("", model.nodes[mu].shape.clone(), model.nodes[mu].dtype)?;
            // grad_mu and grad_sigma are actually the same as x here!
            // they are treated as gradients for the sake of API harmonization
            let grad_mu = step.parameter("", model.nodes[x].shape.clone(), model.nodes[x].dtype)?;
            let old_sigma =
                step.parameter("", model.nodes[mu].shape.clone(), model.nodes[mu].dtype)?;
            let grad_sigma =
                step.parameter("", model.nodes[x].shape.clone(), model.nodes[x].dtype)?;
            let old_params = vec![old_mu, old_sigma];
            let grads = vec![grad_mu, grad_sigma];

            let mut mean = step.reduce_mean(grad_mu, 0, true)?;
            for i in 1..model.nodes[mu].shape.ndims() - 1 {
                mean = step.reduce_mean(mean, i as i64, true)?;
            }
            let mul = step.mul(mom, old_mu)?;
            let mul1 = step.mul(mom1, mean)?;
            let new_mu = step.add(mul, mul1)?;

            let zero_centered = step.sub(grad_mu, mean)?;
            let squared = step.mul(zero_centered, zero_centered)?;
            let mut std = step.reduce_mean(squared, 0, true)?;
            for i in 1..model.nodes[mu].shape.ndims() - 1 {
                std = step.reduce_mean(mean, i as i64, true)?;
            }
            let mul = step.mul(mom, old_sigma)?;
            let mul1 = step.mul(mom1, std)?;
            let new_sigma = step.add(mul, mul1)?;

            let new_params = vec![new_mu, new_sigma];

            Ok::<BatchNormOptimizer, ContextError>(BatchNormOptimizer {
                step,
                old_params,
                grads,
                new_params,
                dummy_state: Vec::new(),
                momentum,
            })
        };

        build().expect("Failed to initialize Batch Normalization optimizer")
    }
}
