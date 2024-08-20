use xla::{Literal, PjRtBuffer, PjRtDevice, PjRtLoadedExecutable};

use crate::graph::{Context, ContextError, Node, NodeIdentifier, Result};

pub struct Metric {
    pub(crate) computation: Context,
    pub(crate) network_outputs: Vec<NodeIdentifier>,
    pub(crate) targets: Vec<NodeIdentifier>,
    pub(crate) loss: NodeIdentifier,
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,
}

pub struct SupervisedModel {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_aux_metrics: usize,

    // forward computation of the network without loss
    pub(crate) network: Context,
    // wraps the node identifiers for the parameters of the network
    // will be buffers at execution
    pub(crate) params: Vec<NodeIdentifier>,
    // list of input nodes
    // will be literals not buffers at executation
    pub(crate) inputs: Vec<NodeIdentifier>,
    // list of output nodes
    // will be buffers at execution
    pub(crate) outputs: Vec<NodeIdentifier>,
    pub(crate) loss: NodeIdentifier,
    pub(crate) targets: Vec<NodeIdentifier>,
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,

    // separate context which takes network outputs and targets
    pub(crate) metric: Metric,
    pub aux_metric_names: Vec<String>,

    // executes the network and gradient metrics
    pub(crate) evaluation_computation: xla::XlaComputation,
    // executes the network and gradient metrics and returns derivatives of the parameters
    pub(crate) gradient_context: Context,
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
        metric: Metric,
        aux_metric_names: Vec<String>,
    ) -> Result<Self> {
        assert_eq!(outputs.len(), metric.network_outputs.len());

        let n_params = params.len();
        let n_inputs = inputs.len();
        let n_outputs = outputs.len();
        let n_targets = metric.targets.len();
        let n_aux_metrics = metric.auxiliary_metrics.len();

        let mut eval_context = network.clone();

        //Fuse compute_metrics to the end of eval_context
        //compute_metrics will take in outputs and targets as inputs
        //outputs is a direct output of inference context
        //targets are supplied in constructor
        let mut remap_nodes = vec![metric.loss];
        remap_nodes.extend(metric.auxiliary_metrics.clone());
        remap_nodes.extend(metric.targets.clone());
        let mut eval_metrics = eval_context.compose_context(
            &metric.computation,
            remap_nodes,
            &outputs,
            &metric.network_outputs,
        )?;

        let evaluation_computation =
            eval_context.build("evaluation_computation", &eval_metrics)?;
        let loss = eval_metrics.drain(0..1).next().unwrap();
        let auxiliary_metrics: Vec<NodeIdentifier> = eval_metrics.drain(0..n_aux_metrics).collect();
        let targets: Vec<NodeIdentifier> = eval_metrics.drain(0..).collect();
        let mut gradient_context = eval_context.clone();

        //Gradient computation: diff loss of eval_context wrt all params
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
            loss,
            targets,
            auxiliary_metrics,
            metric,
            aux_metric_names,
            evaluation_computation,
            gradient_context,
            gradients,
        })
    }

    pub fn compile_inference(
        &mut self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedInferenceExecutable> {
        let n_params = self.n_params;
        let n_inputs = self.n_inputs;
        let n_outputs = self.n_outputs;

        let inference_computation = self.network.build("inference", &self.outputs)?;

        let executable = inference_computation.compile(&client)?;

        let supervised_inf = SupervisedInferenceExecutable {
            n_params,
            n_inputs,
            n_outputs,
            executable,
        };

        Ok(supervised_inf)
    }
    pub fn compile_evaluation(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedEvaluationExecutable> {
        let n_params = self.n_params;
        let n_inputs = self.n_inputs;
        let n_outputs = self.n_outputs;
        let n_targets = self.n_targets;
        let n_metrics = self.metric.auxiliary_metrics.len();

        let executable = self.evaluation_computation.compile(&client)?;

        let supervised_eval = SupervisedEvaluationExecutable {
            n_params,
            n_inputs,
            n_outputs,
            n_targets,
            n_metrics,
            executable,
        };

        Ok(supervised_eval)
    }
    pub fn compile_gradient(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedGradientExecutable> {
        let n_params = self.n_params;
        let n_inputs = self.n_inputs;
        let n_outputs = self.n_outputs;
        let n_targets = self.n_targets;
        let n_metrics = self.metric.auxiliary_metrics.len();

        let executable = self.evaluation_computation.compile(&client)?;

        let supervised_grad = SupervisedGradientExecutable {
            n_params,
            n_inputs,
            n_outputs,
            n_targets,
            n_metrics,
            executable,
        };

        Ok(supervised_grad)
    }
}

pub struct SupervisedInferenceExecutable {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl SupervisedInferenceExecutable {
    pub fn run(
        &self,
        parameters: Vec<PjRtBuffer>,
        inputs: Vec<Literal>,
    ) -> Result<
        // network outputs
        Vec<PjRtBuffer>,
    > {
        let mut input_buff: Vec<PjRtBuffer> = inputs
            .iter()
            .map(|literal| {
                self.executable
                    .client()
                    .buffer_from_host_literal(None, literal)
            })
            .filter(|buff| std::result::Result::is_ok(&buff))
            .map(|buff| buff.unwrap())
            .collect();

        input_buff.extend(parameters.into_iter());

        let res: Vec<PjRtBuffer> = self
            .executable
            .execute_b(&input_buff)?
            .into_iter()
            .flatten()
            .collect::<Vec<PjRtBuffer>>();

        Ok(res)
    }
}

pub struct SupervisedEvaluationExecutable {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_metrics: usize,
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl SupervisedEvaluationExecutable {
    pub fn run(
        &self,
        mut parameters: Vec<PjRtBuffer>,
        mut inputs: Vec<Literal>,
        mut targets: Vec<Literal>,
    ) -> Result<(
        // network outputs
        Vec<PjRtBuffer>,
        // loss
        PjRtBuffer,
        // auxiliary metrics
        Vec<PjRtBuffer>,
    )> {
        let mut input_buffer = Vec::new();
        for param in parameters.drain(0..) {
            input_buffer.push(param);
        }
        for inp in inputs.drain(0..) {
            input_buffer.push(
                self.executable
                    .client()
                    .buffer_from_host_literal(None, &inp)?,
            );
        }
        for tar in targets.drain(0..) {
            input_buffer.push(
                self.executable
                    .client()
                    .buffer_from_host_literal(None, &tar)?,
            );
        }

        let mut res_unsplit: Vec<PjRtBuffer> =
            self.executable.execute_b(&input_buffer)?.pop().unwrap();

        let mut outputs = Vec::new();
        let mut loss = None;
        let mut metrics = Vec::new();
        for (i, x) in res_unsplit.into_iter().enumerate() {
            if i < self.n_outputs {
                outputs.push(x);
            } else if i == self.n_outputs {
                loss = Some(x);
            } else {
                metrics.push(x);
            }
        }

        let loss = loss.unwrap();

        Ok((outputs, loss, metrics))
    }
}

pub struct SupervisedGradientExecutable {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_metrics: usize,
    pub(crate) executable: xla::PjRtLoadedExecutable,
}

impl SupervisedGradientExecutable {
    pub fn run(
        &self,
        mut parameters: Vec<PjRtBuffer>,
        mut inputs: Vec<Literal>,
        mut targets: Vec<Literal>,
    ) -> Result<(
        // network outputs
        Vec<PjRtBuffer>,
        // loss
        PjRtBuffer,
        // auxiliary metrics
        Vec<PjRtBuffer>,
        // gradients
        Vec<PjRtBuffer>,
    )> {
        let mut input_buffer = Vec::new();
        for param in parameters.drain(0..) {
            input_buffer.push(param);
        }
        for inp in inputs.drain(0..) {
            input_buffer.push(
                self.executable
                    .client()
                    .buffer_from_host_literal(None, &inp)?,
            );
        }
        for tar in targets.drain(0..) {
            input_buffer.push(
                self.executable
                    .client()
                    .buffer_from_host_literal(None, &tar)?,
            );
        }

        let res_unsplit: Vec<PjRtBuffer> = self.executable.execute_b(&input_buffer)?.pop().unwrap();

        let mut outputs = Vec::new();
        let mut loss = None;
        let mut metrics = Vec::new();
        let mut grads = Vec::new();
        for (i, x) in res_unsplit.into_iter().enumerate() {
            if i < self.n_outputs {
                outputs.push(x);
            } else if i == self.n_outputs {
                loss = Some(x);
            } else if i < self.n_outputs + 1 + self.n_metrics {
                metrics.push(x);
            } else {
                grads.push(x)
            }
        }

        let loss = loss.unwrap();

        Ok((outputs, loss, metrics, grads))
    }
}
