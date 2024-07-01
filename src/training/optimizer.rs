use crate::graph::context::Result;
use crate::graph::{ContextError, Node};
use crate::graph::{Context, NodeIdentifier};

pub trait Optimizer<const P: usize, const S: usize, U> {
    fn get_step(&self) -> &Context;
    fn get_old_params(&self) -> [NodeIdentifier; P];
    fn get_gradients(&self) -> [NodeIdentifier; P];
    fn get_new_params(&self) -> [NodeIdentifier; P];
    fn get_old_state(&self) -> [NodeIdentifier; S];
    fn get_new_state(&self) -> [NodeIdentifier; S];
    fn get_user_params(&self) -> U;
    fn initialize(user_params: U, model_params: &[NodeIdentifier], model: &Context) -> Self;
}

pub struct ChainedOptimizer<const P: usize, const S: usize, U> {
    step: Context,
    old_params: [NodeIdentifier; P],
    grads: [NodeIdentifier; P],
    new_params: [NodeIdentifier; P],
    old_state: [NodeIdentifier; S],
    new_state: [NodeIdentifier; S],
    pub user_params: U,
}

// WILL FAIL!
// TODO: Need to get node identifiers into the merged context!
/*
pub fn chain<
    const P1: usize,
    const P2: usize,
    const P3: usize,
    const S1: usize,
    const S2: usize,
    const S3: usize,
    U1,
    U2,
    O1: Optimizer<P1, S1, U1>,
    O2: Optimizer<P2, S2, U2>
>(opt1: O1, opt2: O2) -> Result<ChainedOptimizer<P3, S3, (U1, U2)>> {
    Ok(ChainedOptimizer{
        step: opt1.get_step().merge_graphs(opt2.get_step(), &[]),
        old_params: [opt1.get_old_params(), opt2.get_old_params()].concat(),
        grads: [opt1.get_gradients(), opt2.get_gradients()].concat(),
        new_params: [opt1.get_new_params(), opt2.get_new_params()].concat(),
        old_state: [opt1.get_old_state(), opt2.get_old_state()].concat(),
        new_state: [opt1.get_new_state(), opt2.get_new_state()].concat(),
        user_params: (opt1.get_user_params(), opt2.get_user_params())
    })
}
*/

pub struct SGD<const P: usize> {
    step: Context,
    old_params: [NodeIdentifier; P],
    grads: [NodeIdentifier; P],
    new_params: [NodeIdentifier; P],
    pub learning_rate: f32,
}

impl<const P: usize> Optimizer<P, 0, f32> for SGD<P> {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn get_old_params(&self) -> [NodeIdentifier; P] {
        self.old_params
    }
    fn get_gradients(&self) -> [NodeIdentifier; P] {
        self.grads
    }
    fn get_new_params(&self) -> [NodeIdentifier; P] {
        self.new_params
    }
    fn get_old_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn get_new_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn get_user_params(&self) -> f32 {
        self.learning_rate
    }
    fn initialize(
        learning_rate: f32,
        model_params: &[NodeIdentifier],
        model: &Context,
    ) -> SGD<P> {
        let build = || {
            let mut step = Context::new();

            let dtype = model.nodes[model_params[0]].dtype;
            let lr = step.scalar(learning_rate, dtype)?;

            let mut old_params = [model_params[0]; P];
            let mut grads = [model_params[0]; P];
            let mut new_params = [model_params[0]; P];

            for (i, node_id) in model_params.iter().enumerate() {
                let model_param = &model.nodes[*node_id];
                let old_param = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let grad = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                let mul = step.mul(lr, grad)?;
                let new_param = step.sub(old_param, mul)?;
                old_params[i] = old_param;
                grads[i] = grad;
                new_params[i] = new_param;
            }

            Ok::<SGD<P>, ContextError>(SGD {
                step,
                old_params,
                grads,
                new_params,
                learning_rate,
            })
        };

        assert_eq!(model_params.len(), P, "Tried to initialize SGD with wrong number of parameters!");

        build().expect("Failed to initialize SGD")
    }
}

pub struct BatchNormOptimizer {
    step: Context,
    old_params: [NodeIdentifier; 2],
    grads: [NodeIdentifier; 2],
    new_params: [NodeIdentifier; 2],
    pub momentum: f32,
}

impl Optimizer<2, 0, f32> for BatchNormOptimizer {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn get_old_params(&self) -> [NodeIdentifier; 2] {
        self.old_params
    }
    fn get_gradients(&self) -> [NodeIdentifier; 2] {
        self.grads
    }
    fn get_new_params(&self) -> [NodeIdentifier; 2] {
        self.new_params
    }
    fn get_old_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn get_new_state(&self) -> [NodeIdentifier; 0] {
        []
    }
    fn get_user_params(&self) -> f32 {
        self.momentum
    }
    fn initialize(
        momentum: f32,
        batchnorm_params: &[NodeIdentifier],
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

            let old_mu = step.parameter("", model.nodes[mu].shape.clone(), model.nodes[mu].dtype)?;
            // grad_mu and grad_sigma are actually the same as x here!
            // they are treated as gradients for the sake of API harmonization
            let grad_mu = step.parameter("", model.nodes[x].shape.clone(), model.nodes[x].dtype)?;
            let old_sigma = step.parameter("", model.nodes[mu].shape.clone(), model.nodes[mu].dtype)?;
            let grad_sigma = step.parameter("", model.nodes[x].shape.clone(), model.nodes[x].dtype)?;
            let old_params = [old_mu, old_sigma];
            let grads = [grad_mu, grad_sigma];

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

            let new_params = [new_mu, new_sigma];

            Ok::<BatchNormOptimizer, ContextError>(BatchNormOptimizer {
                step,
                old_params,
                grads,
                new_params,
                momentum,
            })
        };

        build().expect("Failed to initialize Batch Normalization optimizer")
    }
}
