use crate::graph::{dtypes::check_int_type, Context, NodeIdentifier, Result};

/// A struct representing the loss and all auxiliary metrics of the network given
/// network outputs and supervised training targets.
///
/// The metric functions defined in this module should be used for building up a context
/// that takes outputs and targets as parameters and that context and its relevant nodes
/// should be bundled using this struct.
pub struct Metrics {
    pub(crate) computation: Context,
    pub(crate) network_outputs: Vec<NodeIdentifier>,
    pub(crate) targets: Vec<NodeIdentifier>,
    pub(crate) loss: NodeIdentifier,
    pub(crate) auxiliary_metrics: Vec<NodeIdentifier>,
}

impl Context {
    /// Accuracy of predictions compared to labels.
    ///
    /// `dense_predictions` is a 2D tensor with first axis being
    /// the batch axis and second being the logits. `sparse_label_vector`
    /// only has the batch axis and should have an integer data type.
    pub fn accuracy(
        &mut self,
        dense_predictions: NodeIdentifier,
        sparse_label_vector: NodeIdentifier,
    ) -> Result<NodeIdentifier> {
        let converted_labels = match check_int_type(self.nodes[sparse_label_vector].dtype) {
            Ok(_) => self.type_cast(sparse_label_vector, xla::ElementType::S64),
            Err(e) => return Err(e),
        };
        let sparse_predictions = self.reduce_argmax(dense_predictions, 1, false)?;
        let compare = self.eq(sparse_predictions, converted_labels)?;
        let converted = self.type_cast(compare, xla::ElementType::F32);
        self.reduce_mean(converted, 0, false)
    }
}
