use xla::{Literal, PjRtBuffer, PjRtClient, PjRtLoadedExecutable, XlaComputation};

use super::optimizers::Optimizer;
use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};
use crate::models::supervised::SupervisedModel;

pub struct SupervisedTrainer<U, O> {
    pub(crate) model: SupervisedModel,
    pub(crate) optimizer: O,
    user_opt_params: U,

    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_metrics: usize,

    // wraps the node identifiers for the parameters of the network
    // will be buffers at execution
    pub(crate) params: Vec<NodeIdentifier>,
    // list of input nodes
    // will be literals not buffers at executation
    pub(crate) inputs: Vec<NodeIdentifier>,
    // list of output nodes
    // will be buffers at execution
    pub(crate) outputs: Vec<NodeIdentifier>,

    // additional inputs to compute_metrics as the targets of the supervised learning algorithm
    pub(crate) targets: Vec<NodeIdentifier>,
    // index into compute_metrics context to find differentiable loss function
    pub(crate) loss: NodeIdentifier,
    // points to additional metrics like accuracy
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,

    old_opt_state: Vec<NodeIdentifier>,
    new_opt_state: Vec<NodeIdentifier>,

    // should take inputs, parameters, targets, optimizer state
    // return outputs, loss, metrics, gradients, updates, new optimizer state, new parameters
    full_pass_comp: XlaComputation,
    full_pass_exec: PjRtLoadedExecutable,
}

pub struct TrainingStepData {
    outputs: Vec<PjRtBuffer>,
    loss: PjRtBuffer,
    metrics: Vec<PjRtBuffer>,
    gradients: Vec<PjRtBuffer>,
    updates: Vec<PjRtBuffer>,
    new_opt_state: Vec<PjRtBuffer>,
    new_params: Vec<PjRtBuffer>,
}

pub struct TrainingHistory {
    loss: Vec<Literal>,
    metrics: Vec<Vec<Literal>>,
    params: Vec<PjRtBuffer>,
}

impl TrainingHistory {
    fn append(
        &mut self,
        mut step_data: TrainingStepData,
    ) -> Result<(
        Vec<PjRtBuffer>,
        Vec<PjRtBuffer>,
        Vec<PjRtBuffer>,
        Vec<PjRtBuffer>,
    )> {
        let loss_literal = step_data.loss.to_literal_sync()?;
        let mut metric_literals = Vec::new();
        for metric in step_data.metrics.drain(0..) {
            metric_literals.push(metric.to_literal_sync()?);
        }
        self.metrics.push(metric_literals);
        self.params = step_data.new_params;
        Ok((
            step_data.outputs,
            step_data.gradients,
            step_data.updates,
            step_data.new_opt_state,
        ))
    }
}

impl<U, O: Optimizer<U>> SupervisedTrainer<U, O> {
    pub fn new(model: SupervisedModel, optimizer: O, client: &PjRtClient) -> Result<Self> {
        let mut full_pass_context = model.network.clone();

        //Fuse compute_metrics to the end of eval_context
        //compute_metrics will take in outputs and targets as inputs
        //outputs is a direct output of inference context
        //targets are supplied in constructor
        let loss_update = full_pass_context.merge_graphs(&model.compute_metrics, &[model.loss])?[0];
        full_pass_context
            .find_and_replace_params(&[("outputs", &model.outputs), ("targets", &model.targets)])?;

        //Gradient computation: diff loss of eval_context wrt all params
        let mut grads = Vec::new();
        for i in 0..model.n_params {
            grads.push(full_pass_context.diff(loss_update, model.params[i])?);
        }

        // STILL NEED TO FUSE WITH OPTIMIZER

        let full_pass_comp = full_pass_context.build("gradient_computation", grads)?;
        let full_pass_exec = full_pass_comp.compile(client)?;

        let user_opt_params = optimizer.get_user_params();
        Ok(SupervisedTrainer {
            model,
            optimizer,
            user_opt_params,
            n_params: model.n_params,
            n_inputs: model.n_inputs,
            n_outputs: model.n_outputs,
            n_targets: model.n_targets,
            n_metrics: model.n_metrics,
        })
    }

    pub fn batch_size(&self) -> usize {
        self.model.network.nodes[self.model.params[0]].shape.sizes[0] as usize
    }

    pub fn step(
        &self,
        mut init_params: Vec<PjRtBuffer>,
        mut train_inputs: Vec<Literal>,
        mut train_targets: Vec<Literal>,
        mut optimizer_state: Vec<PjRtBuffer>,
    ) -> Result<TrainingStepData> {
        assert_eq!(init_params.len(), self.n_params);
        assert_eq!(train_inputs.len(), self.n_inputs);
        assert_eq!(train_targets.len(), self.n_targets);
        assert_eq!(optimizer_state.len(), self.optimizer.state_size());

        let mut all_inputs = Vec::new();
        for param in init_params.drain(0..) {
            all_inputs.push(param);
        }
        for input in train_inputs.drain(0..) {
            all_inputs.push(
                self.full_pass_exec
                    .client()
                    .buffer_from_host_literal(None, &input)?,
            );
        }
        for target in train_targets.drain(0..) {
            all_inputs.push(
                self.full_pass_exec
                    .client()
                    .buffer_from_host_literal(None, &target)?,
            )
        }
        for os in optimizer_state.drain(0..) {
            all_inputs.push(os);
        }

        let mut all_outputs = self.full_pass_exec.execute_b(&all_inputs)?[0];

        assert_eq!(
            all_outputs.len(),
            self.n_outputs + 1 + self.n_metrics + 3 * self.n_params + self.optimizer.state_size()
        );

        let outputs: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_outputs).collect();
        let loss = all_outputs.remove(0);
        let metrics: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_metrics).collect();
        let gradients: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_params).collect();
        let updates: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_params).collect();
        let new_opt_state: Vec<PjRtBuffer> =
            all_outputs.drain(0..self.optimizer.state_size()).collect();
        let new_params: Vec<PjRtBuffer> = all_outputs.drain(0..).collect();

        Ok(TrainingStepData {
            outputs,
            loss,
            metrics,
            gradients,
            updates,
            new_opt_state,
            new_params,
        })
    }

    pub fn fit_infinite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        mut init_params: Vec<Literal>,
        mut train_dataset: D,
        n_steps: usize,
        print_progress: bool,
    ) -> Result<TrainingHistory> {
        let mut params = Vec::new();
        for p in init_params.iter() {
            params.push(
                self.full_pass_exec
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
                self.full_pass_exec
                    .client()
                    .buffer_from_host_literal(None, osh)?,
            );
        }

        for i in 0..n_steps {
            let (mut inps, mut targs) = train_dataset.next().unwrap();

            let step_data = self.step(record.params, inps, targs, opt_state)?;

            let (outputs, gradients, updates, new_opt_state) = record.append(step_data)?;
            opt_state = new_opt_state;
        }

        Ok(record)
    }

    pub fn fit_finite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        client: PjRtClient,
        mut init_params: Vec<Literal>,
        mut train_dataset: fn(usize) -> D,
        n_epochs: usize,
        print_progress: bool,
    ) -> Result<TrainingHistory> {
        let mut params = Vec::new();
        for p in init_params.iter() {
            params.push(
                self.full_pass_exec
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
                self.full_pass_exec
                    .client()
                    .buffer_from_host_literal(None, osh)?,
            );
        }

        for i in 0..n_epochs {
            for (mut inps, mut targs) in train_dataset(i) {
                let step_data = self.step(record.params, inps, targs, opt_state)?;

                let (outputs, gradients, updates, new_opt_state) = record.append(step_data)?;
                opt_state = new_opt_state;
            }
        }

        Ok(record)
    }
}
