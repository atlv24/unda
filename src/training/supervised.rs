use xla::{Literal, PjRtBuffer, PjRtClient, PjRtLoadedExecutable, XlaComputation};

use super::optimizers::Optimizer;
use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};
use crate::models::supervised::SupervisedModel;

pub struct SupervisedTrainer<U, O> {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_aux_metrics: usize,

    pub(crate) model: SupervisedModel,
    pub(crate) optimizer: O,
    user_opt_params: U,

    // should take parameters, inputs, targets, optimizer state
    // return outputs, loss, metrics, gradients, new parameters, new optimizer state
    full_step_ctx: Context,
    full_step_comp: XlaComputation,
    full_step_exec: PjRtLoadedExecutable,

    // points to the optimized parameters as an output of the context
    pub(crate) new_params: Vec<NodeIdentifier>,

    pub(crate) old_opt_state: Vec<NodeIdentifier>,
    pub(crate) new_opt_state: Vec<NodeIdentifier>,
}

pub struct TrainingStepData {
    outputs: Vec<PjRtBuffer>,
    loss: PjRtBuffer,
    metrics: Vec<PjRtBuffer>,
    gradients: Vec<PjRtBuffer>,
    new_params: Vec<PjRtBuffer>,
    new_opt_state: Vec<PjRtBuffer>,
}

pub struct TrainingHistory {
    loss: Vec<Literal>,
    metrics: Vec<Vec<Literal>>,
    params: Vec<PjRtBuffer>,
}

impl TrainingHistory {
    // append loss and metrics while replacing parameters
    // and return the additional data of outputs, gradients, new optimizer state
    fn append(
        &mut self,
        mut step_data: TrainingStepData,
    ) -> Result<(
        Vec<PjRtBuffer>,
        Vec<PjRtBuffer>,
        Vec<PjRtBuffer>,
    )> {
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
    // TODO: WILL FAIL NEED PROPR GRAPH MERGING
    pub fn new(model: SupervisedModel, optimizer: O, client: &PjRtClient) -> Result<Self> {
        let mut full_step_ctx = model.gradient_context.clone();

        // Fuse optimizer to the paramaters and gradients of the network
        let mut net_outputs = model.params.clone();
        net_outputs.extend(model.gradients.clone());
        let mut opt_inputs = optimizer.get_old_params().clone();
        opt_inputs.extend(optimizer.get_gradients());
        let mut to_remap = optimizer.get_old_state().clone();
        to_remap.extend(optimizer.get_new_params());
        to_remap.extend(optimizer.get_new_state());
        let mut remapped = full_step_ctx.compose_context(
            &optimizer.get_step(),
            to_remap,
            &net_outputs,
            &opt_inputs,
        )?;
        let old_opt_state: Vec<NodeIdentifier> = remapped.drain(0..optimizer.state_size()).collect();
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
            new_opt_state
        })
    }

    pub fn batch_size(&self) -> usize {
        self.model.network.nodes[self.model.inputs[0]].shape.sizes[0] as usize
    }

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
            self.n_outputs + 1 + self.n_aux_metrics + 2 * self.n_params + self.optimizer.state_size()
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

    pub fn fit_infinite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        init_params: Vec<Literal>,
        mut train_dataset: D,
        n_steps: usize,
        print_progress: bool,
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

    pub fn fit_finite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        init_params: Vec<Literal>,
        train_dataset: fn(usize) -> D,
        n_epochs: usize,
        print_progress: bool,
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
