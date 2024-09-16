use xla::{Literal, PjRtBuffer};

use crate::graph::{Context, NodeIdentifier, Result};

/// Helper struct for inference.
pub struct InferenceExecutable {
    pub n_params: usize,
    pub n_inputs: usize,
    pub n_outputs: usize,
    pub executable: xla::PjRtLoadedExecutable,
}

impl InferenceExecutable {
    /// Transfer parameters to client device.
    pub fn load_params(&self, parameters: Vec<Literal>) -> Result<Vec<PjRtBuffer>> {
        let param_bufs: Vec<PjRtBuffer> = parameters
            .iter()
            .map(|literal| {
                self.executable
                    .client()
                    .buffer_from_host_literal(None, literal)
            })
            .filter(|buff| std::result::Result::is_ok(&buff))
            .map(|buff| buff.unwrap())
            .collect();
        Ok(param_bufs)
    }

    /// Run the network on parameters and inputs and return all outputs as buffers.
    pub fn run(
        &self,
        parameters: Vec<PjRtBuffer>,
        inputs: Vec<Literal>,
    ) -> Result<Vec<PjRtBuffer>> {
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
