use xla::{ElementType, Literal};

use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};

pub trait Optimizer<U> {
    // The Context should take parameters, gradients, and state in that order
    // should return new parameters, new state in that order
    fn get_step(&self) -> &Context;
    fn n_params(&self) -> usize;
    fn state_size(&self) -> usize;
    fn get_old_params(&self) -> Vec<NodeIdentifier>;
    fn get_gradients(&self) -> Vec<NodeIdentifier>;
    fn get_new_params(&self) -> Vec<NodeIdentifier>;
    fn get_old_state(&self) -> Vec<NodeIdentifier>;
    fn get_new_state(&self) -> Vec<NodeIdentifier>;
    fn get_user_params(&self) -> U;
    fn new(user_params: U, model_params: Vec<NodeIdentifier>, model: &Context) -> Self;
    fn init_state(&self) -> Vec<Literal>;
}

#[derive(Clone)]
pub enum LearningRateSchedule {
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
                    // why clone here???
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
}

impl LRSchedOffset {
    pub fn build_context(&self) -> Result<(NodeIdentifier, Context, NodeIdentifier)> {
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

//*/
pub struct SGD {
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
    fn state_size(&self) -> usize {
        1
    }
    fn get_old_params(&self) -> Vec<NodeIdentifier> {
        self.old_params.clone()
    }
    fn get_gradients(&self) -> Vec<NodeIdentifier> {
        self.grads.clone()
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
    // TODO WILL FAIL NEED PROPER GRAPH MERGING
    fn new(
        lr_schedule: LearningRateSchedule,
        model_params: Vec<NodeIdentifier>,
        model: &Context,
    ) -> SGD {
        let build = || {
            let (iter_node, scheduler, lr_node) = lr_schedule.offset().build_context()?;

            let mut step = Context::new();

            let n_params = model_params.len();

            let dtype = model.nodes[model_params[0]].dtype;

            let mut old_params = Vec::new();
            let mut grads = Vec::new();
            let mut new_params = Vec::new();

            for (i, node_id) in model_params.iter().enumerate() {
                let model_param = &model.nodes[*node_id];
                let old_param = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
                old_params.push(old_param);
            }
            for (i, node_id) in model_params.iter().enumerate() {
                let model_param = &model.nodes[*node_id];
                let grad = step.parameter("", model_param.shape.clone(), model_param.dtype)?;
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
    fn get_old_params(&self) -> Vec<NodeIdentifier> {
        self.old_params.clone()
    }
    fn get_gradients(&self) -> Vec<NodeIdentifier> {
        self.grads.clone()
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
            let old_sigma =
                step.parameter("", model.nodes[mu].shape.clone(), model.nodes[mu].dtype)?;
            let grad_mu = step.parameter("", model.nodes[x].shape.clone(), model.nodes[x].dtype)?;
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

        build().expect("Failed to new Batch Normalization optimizer")
    }
    fn init_state(&self) -> Vec<Literal> {
        return Vec::new();
    }
}
