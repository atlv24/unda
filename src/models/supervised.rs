use xla::{Literal, PjRtBuffer, PjRtDevice, PjRtLoadedExecutable};

use crate::graph::{Context, ContextError, Node, NodeIdentifier, Result};

pub struct SupervisedModel {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub n_targets: usize,
    pub n_metrics: usize,

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

    // separate context which takes parameters, outputs, and targets
    pub(crate) compute_metrics: Context,
    pub(crate) metric_names: Vec<String>,
    // additional inputs to compute_metrics as the targets of the supervised learning algorithm
    pub(crate) targets: Vec<NodeIdentifier>,
    // index into compute_metrics context to find differentiable loss function
    pub(crate) loss: NodeIdentifier,
    // points to additional metrics like accuracy
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,

    // executes the network context without Evaluationuating metrics
    pub(crate) inference_computation: xla::XlaComputation,
    // executes the network and gradient metrics
    pub(crate) evaluation_computation: xla::XlaComputation,
    // executes the network and gradient metrics and returns derivatives of the parameters
    pub(crate) gradient_computation: xla::XlaComputation,
}

impl SupervisedModel {
    // this function should
    // build the inference_computation from the network context
    // fuse the network and compute_metrics contexts and build the evaluation_computation
    // further augment the context to return derivatives of all params and then build the gradient_computation
    pub fn new(
        mut network: Context,
        params: Vec<NodeIdentifier>,
        inputs: Vec<NodeIdentifier>,
        outputs: Vec<NodeIdentifier>,
        compute_metrics: Context,
        metric_names: Vec<String>,
        targets: Vec<NodeIdentifier>,
        loss: NodeIdentifier,
        auxiliary_metrics: Vec<NodeIdentifier>,
    ) -> Result<Self> {
        let n_params = params.len();
        let n_inputs = inputs.len();
        let n_outputs = outputs.len();
        let n_targets = outputs.len();
        let n_metrics = auxiliary_metrics.len();

        let inference_computation = network.build("inference_computation", outputs.clone())?;
        let mut eval_context = network.clone();

        //Fuse compute_metrics to the end of eval_context
        //compute_metrics will take in outputs and targets as inputs
        //outputs is a direct output of inference context
        //targets are supplied in constructor
        let loss_update = eval_context.merge_graphs(&compute_metrics, &[loss])?[0];
        eval_context.find_and_replace_params(&[("outputs", &outputs), ("targets", &targets)])?;

        let evaluation_computation =
            eval_context.build("evaluation_computation", vec![loss_update])?;
        let mut grad_context = eval_context.clone();

        //Gradient computation: diff loss of eval_context wrt all params
        let mut grads = Vec::new();
        for i in 0..n_params {
            grads.push(grad_context.diff(loss_update, params[i])?);
        }

        let gradient_computation = grad_context.build("gradient_computation", grads)?;

        Ok(Self {
            n_params,
            n_inputs,
            n_outputs,
            n_targets,
            n_metrics,
            network,
            params,
            inputs,
            outputs,
            compute_metrics,
            metric_names,
            targets,
            loss: loss_update,
            auxiliary_metrics,
            inference_computation,
            evaluation_computation,
            gradient_computation,
        })
    }

    pub fn compile_inference(
        &self,
        client: xla::PjRtClient,
    ) -> Result<SupervisedInferenceExecutable> {
        let n_params = self.n_params;
        let n_inputs = self.n_inputs;
        let n_outputs = self.n_outputs;

        let executable = self.inference_computation.compile(&client)?;

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
        let n_metrics = self.n_metrics;

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
        let n_metrics = self.n_metrics;

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
