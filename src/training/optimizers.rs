use half::vec;
use smallvec::SmallVec;
use xla::{ElementType, Literal};

use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier, Shape};
use crate::graph::{ContextError, Node};

pub trait Optimizer<U> {
    /// This trait represents an optimizer which takes network parameters,
    /// gradients of those parameters, optional "target" inputs,
    /// and its own state to return new parameters and update its own state.

    /// Returns a context which takes parameters, gradients, optional targets inputs,
    /// and state in that order to return new parameters and new state.
    fn get_step(&self) -> &Context;

    /// Count parameter tensors which the optimizer acts on.
    fn n_params(&self) -> usize;

    /// How many of the targets from the `SupervisedTrainer` will the optimizer take?
    /// Most optimizers will probably not take any targets from the trainer,
    /// but batch normalization with a dynamic batch size will.
    fn n_target_inputs(&self) -> usize;

    /// Number of tensors in the optimizer's state
    fn state_size(&self) -> usize;

    /// Point to the old parameters in the step context.
    fn get_old_params(&self) -> Vec<NodeIdentifier>;

    /// Point to the gradients in the step context.
    fn get_gradients(&self) -> Vec<NodeIdentifier>;

    /// Point to the target inputs in the step context.
    ///
    /// The philosophy here is that training targets could
    /// be used to encode relevant metadata for the optimization
    /// step. In particular, for the dynamic batch normalization
    /// optimizer, the targets can be used to hold the batch mask.
    fn get_target_inputs(&self) -> Vec<NodeIdentifier>;

    /// Point to the new parameters in the step context.
    fn get_new_params(&self) -> Vec<NodeIdentifier>;

    /// Point to the old state in the step context.
    fn get_old_state(&self) -> Vec<NodeIdentifier>;

    /// Point to the new state in the step context.
    fn get_new_state(&self) -> Vec<NodeIdentifier>;

    /// Get any user-specified parameters to the optimizer (such as learning rate or momentum).
    fn get_user_params(&self) -> U;

    /// Create the optimizer from some network parameters and a network context.
    fn new(user_params: U, network_params: Vec<NodeIdentifier>, network: &Context) -> Self;

    /// Initialize the state of the optimizer.
    fn init_state(&self) -> Vec<Literal>;
}

#[derive(Clone)]
pub enum LearningRateSchedule {
    /// This enum specifies the options for learning rate schedules.
    ///
    /// `Constant` will simply give a constant learning rate.
    /// `ExpDecay` starts from `init_lr` and multiplies it be `coefficient`
    /// every `decay_every` optimization steps.
    /// `CosineAnneal` states from `init_lr` and increases to `max_lr`
    /// after `n_steps_to_peak` by following a cosine curve.
    /// `Then` executes `schedule_1` for `n_steps` and then executes `schedule_2`.
    Constant(f32),
    ExpDecay {
        init_lr: f32,
        decay_every: usize,
        coefficient: f32,
    },
    CosineAnneal {
        init_lr: f32,
        max_lr: f32,
        n_steps_to_peak: usize,
    },
    Then {
        schedule_1: Box<LearningRateSchedule>,
        n_steps: usize,
        schedule_2: Box<LearningRateSchedule>,
    },
}

enum LRSchedOffset {
    /// Helper enum for converting user-facing `LearningRateSchedule` to a Context.
    Constant(f32, usize),
    ExpDecay {
        init_lr: f32,
        decay_every: usize,
        coefficient: f32,
        offset: usize,
    },
    CosineAnneal {
        init_lr: f32,
        max_lr: f32,
        n_steps_to_peak: usize,
        offset: usize,
    },
    Then {
        schedule_1: Box<LRSchedOffset>,
        n_steps: usize,
        schedule_2: Box<LRSchedOffset>,
        offset: usize,
    },
}

impl LearningRateSchedule {
    fn offset(&self) -> LRSchedOffset {
        fn zero_offset(sched: &LearningRateSchedule) -> LRSchedOffset {
            match sched {
                &LearningRateSchedule::Constant(lr) => LRSchedOffset::Constant(lr, 0),
                &LearningRateSchedule::ExpDecay {
                    init_lr,
                    decay_every,
                    coefficient,
                } => LRSchedOffset::ExpDecay {
                    init_lr,
                    decay_every,
                    coefficient,
                    offset: 0,
                },
                &LearningRateSchedule::CosineAnneal {
                    init_lr,
                    max_lr,
                    n_steps_to_peak,
                } => LRSchedOffset::CosineAnneal {
                    init_lr,
                    max_lr,
                    n_steps_to_peak,
                    offset: 0,
                },
                LearningRateSchedule::Then {
                    schedule_1,
                    n_steps,
                    schedule_2,
                } => LRSchedOffset::Then {
                    schedule_1: Box::new(zero_offset(schedule_1.as_ref())),
                    n_steps: *n_steps,
                    schedule_2: Box::new(zero_offset(schedule_2.as_ref())),
                    offset: 0,
                },
            }
        }

        fn recurse(sched: &LRSchedOffset, off: usize) -> LRSchedOffset {
            match sched {
                &LRSchedOffset::Constant(lr, offset) => LRSchedOffset::Constant(lr, off + offset),
                &LRSchedOffset::ExpDecay {
                    init_lr,
                    decay_every,
                    coefficient,
                    offset,
                } => LRSchedOffset::ExpDecay {
                    init_lr,
                    decay_every,
                    coefficient,
                    offset: off + offset,
                },
                &LRSchedOffset::CosineAnneal {
                    init_lr,
                    max_lr,
                    n_steps_to_peak,
                    offset,
                } => LRSchedOffset::CosineAnneal {
                    init_lr,
                    max_lr,
                    n_steps_to_peak,
                    offset: off + offset,
                },
                LRSchedOffset::Then {
                    schedule_1,
                    n_steps,
                    schedule_2,
                    offset,
                } => {
                    let offset_1 = recurse(schedule_1.as_ref(), off + offset);
                    let offset_2 = recurse(schedule_2.as_ref(), off + offset);
                    LRSchedOffset::Then {
                        schedule_1: Box::new(offset_1),
                        n_steps: *n_steps,
                        schedule_2: Box::new(offset_2),
                        offset: off + offset,
                    }
                }
            }
        }
        recurse(&zero_offset(self), 0)
    }

    /// From the specific learning rate schedule compute
    /// (node identifier for the step/input,
    /// context for computing the learning rate from the step,
    /// node identifier for the learning rate/output)
    pub fn build_context(&mut self) -> Result<(NodeIdentifier, Context, NodeIdentifier)> {
        self.offset().build_context()
    }
}

impl LRSchedOffset {
    fn build_context(&self) -> Result<(NodeIdentifier, Context, NodeIdentifier)> {
        let mut scheduler = Context::new();
        let iteration = scheduler.parameter("iteration", [], ElementType::U32)?;

        match self {
            &Self::Constant(lr, _) => {
                let lr_node = scheduler.scalar(lr, ElementType::F32)?;
                Ok((iteration, scheduler, lr_node))
            }
            &Self::ExpDecay {
                init_lr,
                decay_every,
                coefficient,
                offset,
            } => {
                let offset_node = scheduler.scalar(offset as u32, ElementType::U32)?;
                let iteration = scheduler.sub(iteration, offset_node)?;
                let dec_ev_node = scheduler.scalar(decay_every as u32, ElementType::U32)?;
                let coeff_node = scheduler.scalar(coefficient, ElementType::F32)?;
                let init_node = scheduler.scalar(init_lr, ElementType::F32)?;
                let iter_div = scheduler.div(iteration, dec_ev_node)?;
                let iter_div_fp = scheduler.type_cast(iter_div, ElementType::F32);
                let decay = scheduler.pow(coeff_node, iter_div_fp)?;
                let lr_node = scheduler.mul(init_node, decay)?;
                Ok((iteration, scheduler, lr_node))
            }
            &Self::CosineAnneal {
                init_lr,
                max_lr,
                n_steps_to_peak,
                offset,
            } => {
                let offset_node = scheduler.scalar(offset as u32, ElementType::U32)?;
                let iteration = scheduler.sub(iteration, offset_node)?;
                let steps_node =
                    scheduler.scalar(2.0 * (n_steps_to_peak as f32) / 3.14159, ElementType::F32)?;
                let max_node = scheduler.scalar(max_lr, ElementType::F32)?;
                let init_node = scheduler.scalar(init_lr, ElementType::F32)?;
                let iter_fp = scheduler.type_cast(iteration, ElementType::F32);
                let iter_div = scheduler.div(iter_fp, steps_node)?;
                let cos_factor = scheduler.cos(iter_div)?;
                let to_add = scheduler.mul(cos_factor, max_node)?;
                let lr_node = scheduler.add(init_node, to_add)?;
                Ok((iteration, scheduler, lr_node))
            }
            Self::Then {
                schedule_1,
                n_steps,
                schedule_2,
                offset,
            } => {
                let (inp_1, mut context_1, lr_1) = schedule_1.as_ref().build_context()?;
                let (inp_2, context_2, lr_2) = schedule_2.as_ref().build_context()?;
                let lr_2 = context_1.combine_graphs(&context_2, &[lr_2])?[0];
                context_1.fuse_nodes(&[(inp_1, inp_2)])?;
                let threshold = context_1.scalar((*n_steps - *offset) as u32, ElementType::U32)?;
                let pred = context_1.ge(inp_1, threshold)?;
                let final_lr = context_1.select(pred, lr_2, lr_1)?;
                Ok((inp_1, context_1, final_lr))
            }
        }
    }
}

pub struct ChainedOptimizer<U> {
    /// This struct represents two different optimizers operating on
    /// two different subsets of network parameters.
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

/// Chain the optimizers `opt1` and `opt2` together.
///
/// The resulting optimizers will expect all the parameters
/// of `opt1` followed by all the parameters of `opt2`.
/// Likewise for the gradients, target inputs, and state.
pub fn chain<U1, U2, O1: Optimizer<U1>, O2: Optimizer<U2>>(
    opt1: O1,
    opt2: O2,
) -> Result<ChainedOptimizer<(U1, U2)>> {
    let mut step = opt1.get_step().clone();
    let mut to_remap = opt2.get_old_params().clone();
    to_remap.extend(opt2.get_gradients());
    to_remap.extend(opt2.get_old_state());
    to_remap.extend(opt2.get_new_params());
    to_remap.extend(opt2.get_new_state());
    let mut remapped = step.combine_graphs(opt2.get_step(), &to_remap)?;
    let old_params2: Vec<NodeIdentifier> = remapped.drain(0..opt2.get_old_params().len()).collect();
    let old_state2: Vec<NodeIdentifier> = remapped.drain(0..opt2.state_size()).collect();
    let gradients2: Vec<NodeIdentifier> = remapped.drain(0..opt2.get_gradients().len()).collect();
    let new_params2: Vec<NodeIdentifier> = remapped.drain(0..opt2.get_new_params().len()).collect();
    let new_state2: Vec<NodeIdentifier> = remapped;

    Ok(ChainedOptimizer {
        step,
        n_params: opt1.n_params() + opt2.n_params(),
        state_size: opt1.state_size() + opt2.state_size(),
        old_params: [opt1.get_old_params().clone(), old_params2].concat(),
        grads: [opt1.get_gradients().clone(), gradients2].concat(),
        new_params: [opt1.get_new_params().clone(), new_params2].concat(),
        old_state: [opt1.get_old_state().clone(), old_state2].concat(),
        new_state: [opt1.get_new_state().clone(), new_state2].concat(),
        user_params: (opt1.get_user_params(), opt2.get_user_params()),
    })
}

pub struct SGD {
    /// This struct represents stochastic gradient descent on a particular set of parameters.
    step: Context,
    n_params: usize,
    old_params: Vec<NodeIdentifier>,
    grads: Vec<NodeIdentifier>,
    new_params: Vec<NodeIdentifier>,
    old_iter: NodeIdentifier,
    new_iter: NodeIdentifier,
    pub lr_schedule: LearningRateSchedule,
}

impl Optimizer<LearningRateSchedule> for SGD {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn n_params(&self) -> usize {
        self.n_params
    }
    fn n_target_inputs(&self) -> usize {
        0
    }
    fn state_size(&self) -> usize {
        1
    }
    fn get_old_params(&self) -> Vec<NodeIdentifier> {
        self.old_params.clone()
    }
    fn get_gradients(&self) -> Vec<NodeIdentifier> {
        self.grads.clone()
    }
    fn get_target_inputs(&self) -> Vec<NodeIdentifier> {
        Vec::new()
    }
    fn get_new_params(&self) -> Vec<NodeIdentifier> {
        self.new_params.clone()
    }
    fn get_old_state(&self) -> Vec<NodeIdentifier> {
        vec![self.old_iter]
    }
    fn get_new_state(&self) -> Vec<NodeIdentifier> {
        vec![self.new_iter]
    }
    fn get_user_params(&self) -> LearningRateSchedule {
        self.lr_schedule.clone()
    }

    fn new(
        lr_schedule: LearningRateSchedule,
        network_params: Vec<NodeIdentifier>,
        network: &Context,
    ) -> SGD {
        let build = || {
            let (iter_node, scheduler, lr_node) = lr_schedule.offset().build_context()?;

            let mut step = Context::new();

            let n_params = network_params.len();

            let dtype = network.nodes[network_params[0]].dtype;

            let mut old_params = Vec::new();
            let mut grads = Vec::new();
            let mut new_params = Vec::new();

            for (i, node_id) in network_params.iter().enumerate() {
                let network_param = &network.nodes[*node_id];
                let old_param =
                    step.parameter("", network_param.shape.clone(), network_param.dtype)?;
                old_params.push(old_param);
            }
            for (i, node_id) in network_params.iter().enumerate() {
                let network_param = &network.nodes[*node_id];
                let grad = step.parameter("", network_param.shape.clone(), network_param.dtype)?;
                grads.push(grad);
            }

            let old_iter = step.parameter("i", [], ElementType::U32)?;
            let one = step.scalar(1, ElementType::U32)?;
            let new_iter = step.add(one, old_iter)?;

            let lr = step.combine_graphs(&scheduler, &[lr_node])?[0];

            for i in 0..n_params {
                let old_param = old_params[i];
                let grad = grads[i];
                let mul = step.mul(lr, grad)?;
                let new_param = step.sub(old_param, mul)?;
                new_params.push(new_param);
            }

            Ok::<SGD, ContextError>(SGD {
                step,
                n_params,
                old_params,
                grads,
                new_params,
                old_iter,
                new_iter,
                lr_schedule,
            })
        };

        build().expect("Failed to new SGD")
    }
    fn init_state(&self) -> Vec<Literal> {
        return Vec::new();
    }
}

pub struct BatchNormOptimizer {
    /// This struct represents a batch normalization optimizer on
    /// the mean and variance of a batchnorm layer.
    ///
    /// This is an atypical concept for a machine learning library.
    /// How does one mesh the typical gradient descent of all parameters of a network
    /// with the online mean/variance estimation of batchnorm?
    /// This is our solution: `BatchNorm` is its own operation which
    /// interacts with autodiff in an unusual way. Specifically, the mean and variance
    /// parameters of the batchnorm layer simply have the inputs to the layer
    /// as their gradients.
    /// This optimizer takes those inputs (as gradients) and computes the rolling mean/variance.
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
    fn n_target_inputs(&self) -> usize {
        0
    }
    fn state_size(&self) -> usize {
        0
    }
    fn get_old_params(&self) -> Vec<NodeIdentifier> {
        self.old_params.clone()
    }
    fn get_gradients(&self) -> Vec<NodeIdentifier> {
        self.grads.clone()
    }
    fn get_target_inputs(&self) -> Vec<NodeIdentifier> {
        Vec::new()
    }
    fn get_new_params(&self) -> Vec<NodeIdentifier> {
        self.new_params.clone()
    }
    fn get_old_state(&self) -> Vec<NodeIdentifier> {
        self.dummy_state.clone()
    }
    fn get_new_state(&self) -> Vec<NodeIdentifier> {
        self.dummy_state.clone()
    }
    fn get_user_params(&self) -> f32 {
        self.momentum
    }
    fn new(
        momentum: f32,
        batchnorm_params: Vec<NodeIdentifier>,
        network: &Context,
    ) -> BatchNormOptimizer {
        let build = || {
            let mut step = Context::new();

            let (mu, sigma, x) = (
                batchnorm_params[0],
                batchnorm_params[1],
                batchnorm_params[2],
            );

            let dtype = network.nodes[mu].dtype;
            let mom = step.scalar(momentum, dtype)?;
            let mut mom1 = step.scalar(1.0, dtype)?;
            mom1 = step.sub(mom1, mom)?;

            let old_mu =
                step.parameter("", network.nodes[mu].shape.clone(), network.nodes[mu].dtype)?;
            // grad_mu and grad_sigma are actually the same as x here!
            // they are treated as gradients for the sake of API harmonization
            let old_sigma = step.parameter(
                "",
                network.nodes[sigma].shape.clone(),
                network.nodes[sigma].dtype,
            )?;
            let grad_mu =
                step.parameter("", network.nodes[x].shape.clone(), network.nodes[x].dtype)?;
            let grad_sigma =
                step.parameter("", network.nodes[x].shape.clone(), network.nodes[x].dtype)?;
            let old_params = vec![old_mu, old_sigma];
            let grads = vec![grad_mu, grad_sigma];

            let mut mean = step.reduce_mean(grad_mu, 0, true)?;
            for i in 1..network.nodes[mu].shape.ndims() - 1 {
                mean = step.reduce_mean(mean, i as i64, true)?;
            }
            let mul = step.mul(mom, old_mu)?;
            let mul1 = step.mul(mom1, mean)?;
            let new_mu = step.add(mul, mul1)?;

            let zero_centered = step.sub(grad_mu, mean)?;
            let squared = step.mul(zero_centered, zero_centered)?;
            let mut std = step.reduce_mean(squared, 0, true)?;
            for i in 1..network.nodes[mu].shape.ndims() - 1 {
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

        build().expect("Failed to new Batch Normalization optimizer")
    }
    fn init_state(&self) -> Vec<Literal> {
        return Vec::new();
    }
}

pub struct DynamicBatchNormOptimizer {
    /// Same idea as above except with a dynamic batch size.
    /// The idea here is simply that a batch mask of floating point 0s
    /// and 1s specifying which passes actually have inputs will
    /// be passed to the SupervisedTrainer. The mean/variance
    /// will be computed accounting for this batch mask,
    step: Context,
    old_params: Vec<NodeIdentifier>,
    grads: Vec<NodeIdentifier>,
    batch_mask: NodeIdentifier,
    new_params: Vec<NodeIdentifier>,
    dummy_state: Vec<NodeIdentifier>,
    pub momentum: f32,
}

impl Optimizer<f32> for DynamicBatchNormOptimizer {
    fn get_step(&self) -> &Context {
        &self.step
    }
    fn n_params(&self) -> usize {
        2
    }
    fn n_target_inputs(&self) -> usize {
        1
    }
    fn state_size(&self) -> usize {
        0
    }
    fn get_old_params(&self) -> Vec<NodeIdentifier> {
        self.old_params.clone()
    }
    fn get_gradients(&self) -> Vec<NodeIdentifier> {
        self.grads.clone()
    }
    fn get_target_inputs(&self) -> Vec<NodeIdentifier> {
        vec![self.batch_mask]
    }
    fn get_new_params(&self) -> Vec<NodeIdentifier> {
        self.new_params.clone()
    }
    fn get_old_state(&self) -> Vec<NodeIdentifier> {
        self.dummy_state.clone()
    }
    fn get_new_state(&self) -> Vec<NodeIdentifier> {
        self.dummy_state.clone()
    }
    fn get_user_params(&self) -> f32 {
        self.momentum
    }
    fn new(
        momentum: f32,
        batchnorm_params: Vec<NodeIdentifier>,
        network: &Context,
    ) -> DynamicBatchNormOptimizer {
        let build = || {
            let mut step = Context::new();

            let (mu, sigma, x) = (
                batchnorm_params[0],
                batchnorm_params[1],
                batchnorm_params[2],
            );

            let dtype = network.nodes[mu].dtype;
            let mom = step.scalar(momentum, dtype)?;
            let mut mom1 = step.scalar(1.0, dtype)?;
            mom1 = step.sub(mom1, mom)?;

            let old_mu =
                step.parameter("", network.nodes[mu].shape.clone(), network.nodes[mu].dtype)?;
            // grad_mu and grad_sigma are actually the same as x here!
            // they are treated as gradients for the sake of API harmonization
            let old_sigma = step.parameter(
                "",
                network.nodes[sigma].shape.clone(),
                network.nodes[sigma].dtype,
            )?;
            let grad_mu =
                step.parameter("", network.nodes[x].shape.clone(), network.nodes[x].dtype)?;
            let grad_sigma =
                step.parameter("", network.nodes[x].shape.clone(), network.nodes[x].dtype)?;
            let old_params = vec![old_mu, old_sigma];
            let grads = vec![grad_mu, grad_sigma];

            let batch_size = network.nodes[x].shape.sizes[0];
            let batch_mask = step.parameter(
                "",
                [batch_size],
                network.nodes[x].dtype,
            )?;
            let mut new_shape = SmallVec::new();
            new_shape.push(batch_size);
            for i in 1..network.nodes[x].shape.ndims() {
                new_shape.push(1);
            }
            let mask_reshaped = step.reshape(batch_mask, Shape{ sizes: new_shape })?;
            let denominator = step.reduce_sum(batch_mask, 0, false)?;

            let grad_mu_masked = step.mul(mask_reshaped, grad_mu)?;

            let mut mean = step.reduce_sum(grad_mu_masked, 0, true)?;
            mean = step.div(mean, denominator)?;
            for i in 1..network.nodes[mu].shape.ndims() - 1 {
                mean = step.reduce_mean(mean, i as i64, true)?;
            }
            let mul = step.mul(mom, old_mu)?;
            let mul1 = step.mul(mom1, mean)?;
            let new_mu = step.add(mul, mul1)?;

            let zero_centered = step.sub(grad_mu, mean)?;
            let zero_centered_masked = step.mul(mask_reshaped, zero_centered)?;
            let squared = step.mul(zero_centered_masked, zero_centered_masked)?;
            let mut std = step.reduce_sum(squared, 0, true)?;
            std = step.div(std, denominator)?;
            for i in 1..network.nodes[mu].shape.ndims() - 1 {
                std = step.reduce_mean(mean, i as i64, true)?;
            }
            let mul = step.mul(mom, old_sigma)?;
            let mul1 = step.mul(mom1, std)?;
            let new_sigma = step.add(mul, mul1)?;

            let new_params = vec![new_mu, new_sigma];

            Ok::<DynamicBatchNormOptimizer, ContextError>(DynamicBatchNormOptimizer {
                step,
                old_params,
                grads,
                batch_mask,
                new_params,
                dummy_state: Vec::new(),
                momentum,
            })
        };

        build().expect("Failed to new Batch Normalization optimizer")
    }
    fn init_state(&self) -> Vec<Literal> {
        return Vec::new();
    }
}
