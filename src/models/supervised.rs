use xla::{Literal, PjRtBuffer};

use crate::graph::{Context, NodeIdentifier, Result};
use crate::models::{inference::InferenceExecutable, metrics::Metrics};

pub struct SupervisedModel {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_aux_metrics: usize,

    /// Forward computation of the network without loss.
    pub(crate) network: Context,
    /// Wraps the node identifiers for the parameters of the network.
    /// These will be buffers at execution
    pub(crate) params: Vec<NodeIdentifier>,
    /// List of input nodes.
    /// These will be literals not buffers at execution.
    pub(crate) inputs: Vec<NodeIdentifier>,
    /// List of output nodes.
    /// These will be buffers at execution.
    pub(crate) outputs: Vec<NodeIdentifier>,

    /// Context which performs a forward pass of the network and
    /// computes metrics based on targets. All node identifiers
    /// for `network` should work the same for `evaluation_context`,
    pub(crate) evaluation_context: Context,
    /// Pre-compiled version of the above computation.
    pub(crate) evaluation_computation: xla::XlaComputation,
    /// Optional names of the auxiliary metrics for printing.
    pub aux_metric_names: Vec<String>,

    /// Points to the loss in `evaluation_context`.
    pub(crate) loss: NodeIdentifier,
    /// Points to the targets in `evaluation_context`.
    pub(crate) targets: Vec<NodeIdentifier>,
    /// Points to auxiliary_metrics in `evaluation_context`.
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,

    /// Executes forward pass of the network, compute loss and metrics,
    /// and compute gradients of the parameters with respect to loss.
    /// All node identifiers which work for `evaluation_context` should work
    /// for `gradient_context`.
    pub(crate) gradient_context: Context,
    /// Point to the gradient nodes in `gradient_context`.
    pub(crate) gradients: Vec<NodeIdentifier>,
}

impl SupervisedModel {
    // this function should
    // build the inference_computation from the network context
    // fuse the network and compute_metrics contexts and build the evaluation_computation
    // further augment the context to return derivatives of all params and then build the gradient_computation
    pub fn new(
        network: Context,
        params: Vec<NodeIdentifier>,
        inputs: Vec<NodeIdentifier>,
        outputs: Vec<NodeIdentifier>,
        metrics: Metrics,
        aux_metric_names: Vec<String>,
    ) -> Result<Self> {
        assert_eq!(outputs.len(), metrics.network_outputs.len());

        let n_params = params.len();
        let n_inputs = inputs.len();
        let n_outputs = outputs.len();
        let n_targets = metrics.targets.len();
        let n_aux_metrics = metrics.auxiliary_metrics.len();

        let mut evaluation_context = network.clone();

        // Fuse compute_metrics to the end of evaluation_context
        // compute_metrics will take in outputs and targets as inputs
        // outputs is a direct output of inference context
        // targets are supplied in constructor
        let mut remap_nodes = vec![metrics.loss];
        remap_nodes.extend(metrics.auxiliary_metrics.clone());
        remap_nodes.extend(metrics.targets.clone());
        let mut eval_metrics = evaluation_context.compose_context(
            &metrics.computation,
            remap_nodes,
            &outputs,
            &metrics.network_outputs,
        )?;

        let evaluation_computation =
            evaluation_context.build("evaluation_computation", &eval_metrics)?;
        let loss = eval_metrics.drain(0..1).next().unwrap();
        let auxiliary_metrics: Vec<NodeIdentifier> = eval_metrics.drain(0..n_aux_metrics).collect();
        let targets: Vec<NodeIdentifier> = eval_metrics.drain(0..).collect();
        let mut gradient_context = evaluation_context.clone();

        //Gradient computation: diff loss of evaluation_context wrt all params
        let mut gradients = Vec::new();
        for i in 0..n_params {
            gradients.push(gradient_context.diff(loss, params[i])?);
        }

        Ok(Self {
            n_params,
            n_inputs,
            n_outputs,
            n_targets,
            n_aux_metrics,
            network,
            params,
            inputs,
            outputs,
            evaluation_context,
            evaluation_computation,
            loss,
            targets,
            auxiliary_metrics,
            aux_metric_names,
            gradient_context,
            gradients,
        })
    }

    pub fn compile_inference(&mut self, client: xla::PjRtClient) -> Result<InferenceExecutable> {
        let n_params = self.n_params;
        let n_inputs = self.n_inputs;
        let n_outputs = self.n_outputs;

        let inference_computation = self.network.build("inference", &self.outputs)?;

        let executable = inference_computation.compile(&client)?;

        let supervised_inf = InferenceExecutable {
            n_params,
            n_inputs,
            n_outputs,
            executable,
        };

        Ok(supervised_inf)
    }
}
