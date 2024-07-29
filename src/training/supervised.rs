use xla::{Literal, PjRtBuffer, PjRtClient, XlaComputation};

use super::optimizer::Optimizer;
use crate::graph::context::Result;
use crate::graph::{Context, NodeIdentifier};
use crate::graph::{ContextError, Node};
use crate::models::supervised::SupervisedModel;

struct SupervisedTrainerStaticBatch<U, O> {
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
    // return outputs, loss, metrics, new optimizer state, new parameters
    full_pass: XlaComputation,
}

impl<U, O: Optimizer<U>> SupervisedTrainerStaticBatch<U, O> {
    pub fn new(model: SupervisedModel, optimizer: O) -> Result<Self> {
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

        let full_pass = full_pass_context.build("gradient_computation", grads)?;

        let user_opt_params = optimizer.get_user_params();
        Ok(SupervisedTrainerStaticBatch {
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

    pub fn train_infinite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        client: PjRtClient,
        mut init_params: Vec<Literal>,
        mut train_dataset: D,
        n_steps: usize,
        print_progress: bool,
    ) -> Result<(
        // loss history
        Vec<Literal>,
        // auxillary metric history
        Vec<Vec<Literal>>,
        // final network params
        Vec<Literal>,
    )> {
        assert_eq!(self.n_params, init_params.len());

        let executable = self.full_pass.compile(&client)?;

        let mut parameters = Vec::new();
        for param in init_params.drain(0..) {
            parameters.push(client.buffer_from_host_literal(None, &param)?)
        }

        let mut opt_state_host = self.optimizer.init_state();
        let mut opt_state = Vec::new();
        for osh in opt_state_host.drain(0..) {
            opt_state.push(client.buffer_from_host_literal(None, &osh)?);
        }

        let mut record_loss = Vec::new();
        let mut record_metrics = Vec::new();

        for i in 0..n_steps {
            let (mut inps, mut targs) = train_dataset.next().unwrap();

            assert_eq!(self.n_inputs, inps.len());
            assert_eq!(self.n_targets, targs.len());

            let mut all_inputs = Vec::new();
            for param in parameters.drain(0..) {
                all_inputs.push(param)
            }
            for inp in inps.drain(0..) {
                all_inputs.push(client.buffer_from_host_literal(None, &inp)?);
            }
            for targ in targs.drain(0..) {
                all_inputs.push(client.buffer_from_host_literal(None, &targ)?);
            }
            for os in opt_state.drain(0..) {
                all_inputs.push(os);
            }

            let mut all_outputs = executable.execute_b(&all_inputs)?.pop().unwrap();
            let network_outputs: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_outputs).collect();
            let loss = all_outputs.pop().unwrap();
            let mut metrics: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_metrics).collect();
            opt_state = all_outputs.drain(0..self.optimizer.state_size()).collect();
            parameters = all_outputs;

            let loss_host = loss.to_literal_sync()?;
            let mut metrics_host = Vec::new();
            for metric in metrics.drain(0..) {
                metrics_host.push(metric.to_literal_sync()?);
            }

            record_loss.push(loss_host);
            record_metrics.push(metrics_host)
        }

        let mut parameters_host = Vec::new();
        for param in parameters {
            parameters_host.push(param.to_literal_sync()?);
        }

        Ok((record_loss, record_metrics, parameters_host))
    }

    pub fn train_finite_data<D: Iterator<Item = (Vec<Literal>, Vec<Literal>)>>(
        &self,
        client: PjRtClient,
        mut init_params: Vec<Literal>,
        mut train_dataset: fn(usize) -> D,
        n_epochs: usize,
        print_progress: bool,
    ) -> Result<(
        // loss history
        Vec<Literal>,
        // auxillary metric history
        Vec<Vec<Literal>>,
        // final network params
        Vec<Literal>,
    )> {
        assert_eq!(self.n_params, init_params.len());

        let executable = self.full_pass.compile(&client)?;

        let mut parameters = Vec::new();
        for param in init_params.drain(0..) {
            parameters.push(client.buffer_from_host_literal(None, &param)?)
        }

        let mut opt_state_host = self.optimizer.init_state();
        let mut opt_state = Vec::new();
        for osh in opt_state_host.drain(0..) {
            opt_state.push(client.buffer_from_host_literal(None, &osh)?);
        }

        let mut record_loss = Vec::new();
        let mut record_metrics = Vec::new();

        for i in 0..n_epochs {
            for (mut inps, mut targs) in train_dataset(i) {
                assert_eq!(self.n_inputs, inps.len());
                assert_eq!(self.n_targets, targs.len());

                let mut all_inputs = Vec::new();
                for param in parameters.drain(0..) {
                    all_inputs.push(param)
                }
                for inp in inps.drain(0..) {
                    all_inputs.push(client.buffer_from_host_literal(None, &inp)?);
                }
                for targ in targs.drain(0..) {
                    all_inputs.push(client.buffer_from_host_literal(None, &targ)?);
                }
                for os in opt_state.drain(0..) {
                    all_inputs.push(os);
                }

                let mut all_outputs = executable.execute_b(&all_inputs)?.pop().unwrap();
                let network_outputs: Vec<PjRtBuffer> =
                    all_outputs.drain(0..self.n_outputs).collect();
                let loss = all_outputs.pop().unwrap();
                let mut metrics: Vec<PjRtBuffer> = all_outputs.drain(0..self.n_metrics).collect();
                opt_state = all_outputs.drain(0..self.optimizer.state_size()).collect();
                parameters = all_outputs;

                let loss_host = loss.to_literal_sync()?;
                let mut metrics_host = Vec::new();
                for metric in metrics.drain(0..) {
                    metrics_host.push(metric.to_literal_sync()?);
                }

                record_loss.push(loss_host);
                record_metrics.push(metrics_host)
            }
        }

        let mut parameters_host = Vec::new();
        for param in parameters {
            parameters_host.push(param.to_literal_sync()?);
        }

        Ok((record_loss, record_metrics, parameters_host))
    }
}
