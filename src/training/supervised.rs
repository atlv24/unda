use xla::{Literal, PjRtBuffer, PjRtClient, PjRtLoadedExecutable, XlaComputation};

use super::optimizers::Optimizer;
use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};
use crate::models::supervised::SupervisedModel;

/// A struct for abstracting the supervised training of a network.
///
/// `U` is the user parameters of the optimizer used for training.
/// `O` is the optimizer type.
pub struct SupervisedTrainer<U, O> {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_aux_metrics: usize,

    /// Supervised model which the optimizer acts on.
    pub(crate) model: SupervisedModel,
    /// Optimizer which acts on the supervised model.
    pub(crate) optimizer: O,
    /// Placeholder for user-specified optimization params.
    user_opt_params: U,

    /// A Context which takes parameters, inputs, targets, optimizer state
    /// and returns outputs, loss, metrics, gradients, new parameters, new optimizer state
    full_step_ctx: Context,
    /// Compiled version of the above Context.
    full_step_comp: XlaComputation,
    /// Device-loaded version of the above computation.
    full_step_exec: PjRtLoadedExecutable,

    /// Points to the optimized parameters in `full_step_ctx`.
    pub(crate) new_params: Vec<NodeIdentifier>,

    /// Points to the old optimizer state in `full_step_ctx`.
    pub(crate) old_opt_state: Vec<NodeIdentifier>,
    /// Points to the new optimizer state in `full_step_ctx`.
    pub(crate) new_opt_state: Vec<NodeIdentifier>,
}

/// Encapsulates the data of a single training step.
pub struct TrainingStepData {
    outputs: Vec<PjRtBuffer>,
    loss: PjRtBuffer,
    metrics: Vec<PjRtBuffer>,
    gradients: Vec<PjRtBuffer>,
    new_params: Vec<PjRtBuffer>,
    new_opt_state: Vec<PjRtBuffer>,
}

/// Encapsulates the history of the loss and all auxiliary metrics, but only the most recent parameters.
pub struct TrainingHistory {
    loss: Vec<Literal>,
    metrics: Vec<Vec<Literal>>,
    params: Vec<PjRtBuffer>,
}

pub enum TrainingLog {
    Mute,
    Terminal,
    LogFile(String),
}

impl TrainingHistory {
    /// Append loss and metrics while replacing parameters
    /// and return the additional data of outputs, gradients, new optimizer state.
    fn append(
        &mut self,
        mut step_data: TrainingStepData,
    ) -> Result<(Vec<PjRtBuffer>, Vec<PjRtBuffer>, Vec<PjRtBuffer>)> {
        let loss_literal = step_data.loss.to_literal_sync()?;
        let mut metric_literals = Vec::new();
        for metric in step_data.metrics.drain(0..) {
            metric_literals.push(metric.to_literal_sync()?);
        }
        self.loss.push(loss_literal);
        self.metrics.push(metric_literals);
        self.params = step_data.new_params;
        Ok((
            step_data.outputs,
            step_data.gradients,
            step_data.new_opt_state,
        ))
    }
}

impl<U, O: Optimizer<U>> SupervisedTrainer<U, O> {
    /// Construct a new SupervisedTrainer from a model and an optimizer
    ///
    /// `optimizer_target_inputs` points to targets in `model.gradient_context`
    /// which are relevant to the optimizer. For example, a floating point
    /// mask of 1s and 0s could be passed as a target to the model, which
    /// can then be used for optimization with a DynamicBatchNorm optimizer.
    pub fn new(
        model: SupervisedModel,
        optimizer: O,
        optimizer_target_inputs: Vec<NodeIdentifier>,
        client: &PjRtClient,
    ) -> Result<Self> {
        let mut full_step_ctx = model.gradient_context.clone();

        // The data from the forward/backward pass which will be passed to the optimizer
        let mut net_outputs = model.params.clone();
        net_outputs.extend(model.gradients.clone());
        // and the relevant targets will be passed to the optimizer
        net_outputs.extend(optimizer_target_inputs);

        // Inputs to the optimizer
        let mut opt_inputs = optimizer.get_old_params().clone();
        opt_inputs.extend(optimizer.get_gradients());
        opt_inputs.extend(optimizer.get_target_inputs());

        // Fuse the optimizer to the forward/backward pass
        let mut to_remap = optimizer.get_old_state().clone();
        to_remap.extend(optimizer.get_new_params());
        to_remap.extend(optimizer.get_new_state());
        let mut remapped = full_step_ctx.compose_context(
            &optimizer.get_step(),
            to_remap,
            &net_outputs,
            &opt_inputs,
        )?;

        // Separate the relevant optimizer node identifiers
        let old_opt_state: Vec<NodeIdentifier> =
            remapped.drain(0..optimizer.state_size()).collect();
        let new_params: Vec<NodeIdentifier> = remapped.drain(0..model.n_params).collect();
        let new_opt_state: Vec<NodeIdentifier> = remapped.drain(0..).collect();

        let mut returns = model.outputs.clone();
        returns.push(model.loss);
        returns.extend(model.auxiliary_metrics.clone());
        returns.extend(model.gradients.clone());
        returns.extend(new_params.clone());
        returns.extend(new_opt_state.clone());

        let full_step_comp = full_step_ctx.build("full_step", &returns)?;
        let full_step_exec = full_step_comp.compile(client)?;

        let user_opt_params = optimizer.get_user_params();
        Ok(SupervisedTrainer {
            n_params: model.n_params,
            n_inputs: model.n_inputs,
            n_outputs: model.n_outputs,
            n_targets: model.n_targets,
            n_aux_metrics: model.n_aux_metrics,
            model,
            optimizer,
            user_opt_params,
            full_step_ctx,
            full_step_comp,
            full_step_exec,
            new_params,
            old_opt_state,
            new_opt_state,
        })
    }

    pub fn get_user_opt_params(&self) -> &U {
        &self.user_opt_params
    }

    pub fn recompile(&mut self, client: &PjRtClient) -> Result<()> {
        self.full_step_exec = self.full_step_comp.compile(client)?;
        Ok(())
    }

    pub fn load_params(&self, mut params: Vec<Literal>) -> Result<Vec<PjRtBuffer>> {
        let mut param_bufs: Vec<PjRtBuffer> = Vec::new();
        for param in params.drain(0..) {
            param_bufs.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, &param)?,
            );
        }
        Ok(param_bufs)
    }

    pub fn batch_size(&self) -> usize {
        self.model.network.nodes[self.model.inputs[0]].shape.sizes[0] as usize
    }

    /// Execute a single training step and return the step data.
    pub fn step(
        &self,
        init_params: &Vec<PjRtBuffer>,
        mut train_inputs: Vec<Literal>,
        mut train_targets: Vec<Literal>,
        optimizer_state: Vec<PjRtBuffer>,
    ) -> Result<TrainingStepData> {
        assert_eq!(init_params.len(), self.n_params);
        assert_eq!(train_inputs.len(), self.n_inputs);
        assert_eq!(train_targets.len(), self.n_targets);
        assert_eq!(optimizer_state.len(), self.optimizer.state_size());

        let mut all_inputs: Vec<&PjRtBuffer> = Vec::new();
        for param in init_params.iter() {
            all_inputs.push(param);
        }
        let mut input_bufs: Vec<PjRtBuffer> = Vec::new();
        for input in train_inputs.drain(0..) {
            input_bufs.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, &input)?,
            );
        }
        for input in input_bufs.iter() {
            all_inputs.push(input);
        }
        let mut target_bufs: Vec<PjRtBuffer> = Vec::new();
        for target in train_targets.drain(0..) {
            target_bufs.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, &target)?,
            )
        }
        for target in target_bufs.iter() {
            all_inputs.push(target);
        }
        for os in optimizer_state.iter() {
            all_inputs.push(os);
        }

        let mut all_outputs = self.full_step_exec.execute_b(&all_inputs)?;
        let mut all_outputs = all_outputs.pop().unwrap();

        assert_eq!(
            all_outputs.len(),
            self.n_outputs
                + 1
                + self.n_aux_metrics
                + 2 * self.n_params
                + self.optimizer.state_size()
        );

        let outputs: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_outputs).collect();
        let loss = all_outputs.remove(0);
        let metrics: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_aux_metrics).collect();
        let gradients: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_params).collect();
        let new_opt_state: Vec<PjRtBuffer> =
            all_outputs.drain(0..self.optimizer.state_size()).collect();
        let new_params: Vec<PjRtBuffer> = all_outputs.drain(0..).collect();

        Ok(TrainingStepData {
            outputs,
            loss,
            metrics,
            gradients,
            new_opt_state,
            new_params,
        })
    }

    /// Fit an infinite iterator of training data.
    ///
    /// `init_params` are the initialized literal parameters of the network.
    /// `train_dataset` is an infinite iterator of training data.
    /// `n_steps` is the number of gradient steps training should last for.
    /// `valid_dataset` is an optional finite validation dataset.
    /// `validate_every` if `validate_dataset`` is not None, validated every this many training steps.
    /// `training_log` whether to print training progress, and whether to send it to a file or the terminal.
    //TODO: actually implement validation and logging.
    pub fn fit_infinite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        init_params: Vec<Literal>,
        mut train_dataset: D,
        n_steps: usize,
        valid_dataset: Option<D>,
        validate_every: usize,
        training_log: TrainingLog,
    ) -> Result<TrainingHistory> {
        let mut params = Vec::new();
        for p in init_params.iter() {
            params.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, p)?,
            )
        }

        let mut record = TrainingHistory {
            loss: Vec::new(),
            metrics: Vec::new(),
            params,
        };

        let opt_state_host = self.optimizer.init_state();
        let mut opt_state = Vec::new();
        for osh in opt_state_host.iter() {
            opt_state.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, osh)?,
            );
        }

        for i in 0..n_steps {
            let (inps, targs) = train_dataset.next().unwrap();

            let step_data = self.step(&record.params, inps, targs, opt_state)?;

            let (_, _, new_opt_state) = record.append(step_data)?;
            opt_state = new_opt_state;
        }

        Ok(record)
    }

    /// Fit a finite iterator of training data.
    ///
    /// `init_params` are the initialized literal parameters of the network.
    /// `train_dataset` is a function which takes the current epoch and returns a shuffled training dataset.
    /// `n_epochs` is the number of times to loop through the training dataset
    /// `valid_dataset` is an optional finite validation dataset.
    /// `validate_every` if `validate_dataset`` is not None, validated every this many epochs.
    /// `training_log` whether to print training progress, and whether to send it to a file or the terminal.
    //TODO: actually implement validation and logging.
    pub fn fit_finite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        init_params: Vec<Literal>,
        train_dataset: fn(usize) -> D,
        n_epochs: usize,
        valid_dataset: Option<D>,
        validate_every: usize,
        training_log: TrainingLog,
    ) -> Result<TrainingHistory> {
        let mut params = Vec::new();
        for p in init_params.iter() {
            params.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, p)?,
            )
        }

        let mut record = TrainingHistory {
            loss: Vec::new(),
            metrics: Vec::new(),
            params,
        };

        let opt_state_host = self.optimizer.init_state();
        let mut opt_state = Vec::new();
        for osh in opt_state_host.iter() {
            opt_state.push(
                self.full_step_exec
                    .client()
                    .buffer_from_host_literal(None, osh)?,
            );
        }

        for i in 0..n_epochs {
            for (inps, targs) in train_dataset(i) {
                let step_data = self.step(&record.params, inps, targs, opt_state)?;

                let (_, _, new_opt_state) = record.append(step_data)?;
                opt_state = new_opt_state;
            }
        }

        Ok(record)
    }
}
