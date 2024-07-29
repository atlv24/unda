use std::collections::HashMap;

use super::NodeIdentifier;
use super::Operation;
use super::Result;

use super::Context;

impl Context {
    pub fn merge_graphs(
        &mut self,
        other: &Context,
        desired_remaps: &[NodeIdentifier],
    ) -> Result<Vec<NodeIdentifier>> {
        let mut old_to_new: HashMap<NodeIdentifier, NodeIdentifier> = HashMap::new();
        let mut addition_queue = self.inputs();

        while let Some(old_node) = addition_queue.pop() {
            let new_id = self.nodes.insert(other.nodes[old_node].clone());

            match self.nodes[new_id].operation.clone() {
                Operation::Add(a, b) => {
                    self.nodes[new_id].operation = Operation::Add(old_to_new[&a], old_to_new[&b])
                }
                Operation::RngUniform(a, b, shape) => {
                    self.nodes[new_id].operation = Operation::RngUniform(old_to_new[&a], old_to_new[&b], shape)
                }
                Operation::Pow(a, b) => {
                    self.nodes[new_id].operation = Operation::Pow(old_to_new[&a], old_to_new[&b])
                }
                Operation::Sub(a, b) => {
                    self.nodes[new_id].operation = Operation::Sub(old_to_new[&a], old_to_new[&b])
                }
                Operation::Mul(a, b) => {
                    self.nodes[new_id].operation = Operation::Mul(old_to_new[&a], old_to_new[&b])
                }
                Operation::MatMul(a, b) => {
                    self.nodes[new_id].operation = Operation::MatMul(old_to_new[&a], old_to_new[&b])
                }
                Operation::Div(a, b) => {
                    self.nodes[new_id].operation = Operation::Div(old_to_new[&a], old_to_new[&b])
                }
                Operation::GreaterThan(a, b) => {
                    self.nodes[new_id].operation = Operation::GreaterThan(old_to_new[&a], old_to_new[&b])
                }
                Operation::GreaterThanEq(a, b) => {
                    self.nodes[new_id].operation = Operation::GreaterThanEq(old_to_new[&a], old_to_new[&b])
                }
                Operation::Equal(a, b) => {
                    self.nodes[new_id].operation = Operation::Equal(old_to_new[&a], old_to_new[&b])
                }
                Operation::NotEqual(a, b) => {
                    self.nodes[new_id].operation = Operation::NotEqual(old_to_new[&a], old_to_new[&b])
                }
                Operation::LessThan(a, b) => {
                    self.nodes[new_id].operation = Operation::LessThan(old_to_new[&a], old_to_new[&b])
                }
                Operation::LessThanEq(a, b) => {
                    self.nodes[new_id].operation = Operation::LessThanEq(old_to_new[&a], old_to_new[&b])
                }
                Operation::StopGradient(a) => {
                    self.nodes[new_id].operation = Operation::StopGradient(old_to_new[&a])
                }
                Operation::Neg(a) => {
                    self.nodes[new_id].operation = Operation::Neg(old_to_new[&a])
                }
                Operation::Exp(a) => {
                    self.nodes[new_id].operation = Operation::Exp(old_to_new[&a])
                }
                Operation::Log(a) => {
                    self.nodes[new_id].operation = Operation::Log(old_to_new[&a])
                }
                Operation::ZerosLike(a) => {
                    self.nodes[new_id].operation = Operation::ZerosLike(old_to_new[&a])
                }
                Operation::OneHot(node) => {
                    self.nodes[new_id].operation = Operation::OneHot(old_to_new[&node])
                }
                Operation::TypeCast(node, t) => {
                    self.nodes[new_id].operation = Operation::TypeCast(old_to_new[&node], t)
                }
                Operation::Reshape(node) => {
                    self.nodes[new_id].operation = Operation::Reshape(old_to_new[&node])
                }
                Operation::Select {
                    pred,
                    on_false,
                    on_true,
                } => {
                    self.nodes[new_id].operation = Operation::Select { pred: old_to_new[&pred], on_true: old_to_new[&on_true], on_false: old_to_new[&on_false] }
                }
                Operation::ReduceMax { node, dim } => {
                    self.nodes[new_id].operation = Operation::ReduceMax { node: old_to_new[&node], dim }
                }
                Operation::ReduceArgmax {
                    node,
                    dim,
                } => {
                    self.nodes[new_id].operation = Operation::ReduceArgmax { node: old_to_new[&node], dim }
                }
                Operation::ReduceSum { node, dim } => {
                    self.nodes[new_id].operation = Operation::ReduceSum { node: old_to_new[&node], dim }
                }
                Operation::ReduceMean { node, dim } => {
                    self.nodes[new_id].operation = Operation::ReduceMean { node: old_to_new[&node], dim }
                }
                Operation::Transpose(a, dim) => {
                    self.nodes[new_id].operation = Operation::Transpose(old_to_new[&a], dim)
                }
                Operation::SliceInDim {
                    node,
                    start,
                    stop,
                    stride,
                    dim,
                } => {
                    self.nodes[new_id].operation = Operation::SliceInDim { node: old_to_new[&node], start, stop, stride, dim }
                }
                Operation::TileInDim { node, n_tiles, dim } => {
                    self.nodes[new_id].operation = Operation::TileInDim { node: old_to_new[&node], n_tiles, dim }
                }
                Operation::RngNormal(a, b, shape) => {
                    self.nodes[new_id].operation = Operation::RngNormal(old_to_new[&a], old_to_new[&b], shape)
                }
                _ => {} //Constants, parameters, don't need nodeid replacement
            }

            match self.nodes[new_id].operation {
                Operation::Constant(_) => self.constants.push(new_id),
                Operation::Parameter(_) => self.parameters.push(new_id),
                _ => (),
            }

            if let Some(deps) = other.dependent_nodes.get(&old_node) {
                for node in deps {
                    addition_queue.insert(0, *node);
                }
            }

            old_to_new.insert(old_node, new_id);
        }

        for (old_node, old_deps) in other.dependent_nodes.clone() {
            let new_node = old_to_new[&old_node];
            let new_deps = old_deps
                .iter()
                .map(|old| old_to_new[old])
                .collect::<Vec<NodeIdentifier>>();

            self.dependent_nodes.insert(new_node, new_deps);
        }

        let mut new_remaps = vec![];

        for old in desired_remaps {
            new_remaps.push(old_to_new[old])
        }

        Ok(new_remaps)
    }

    pub fn find_and_replace_params(
        &mut self,
        param_reps: &[(&str, &[NodeIdentifier])],
    ) -> Result<()> {
        for (param_name, rep_with) in param_reps {
            let params_with_name: Vec<NodeIdentifier> = self
                .nodes
                .clone()
                .into_iter()
                .filter(|(_, node)| match node.operation.clone() {
                    Operation::Parameter(name) => name.contains(param_name),
                    _ => false,
                })
                .map(|(id, _)| id)
                .collect();

            if params_with_name.len() != rep_with.len() {
                return Err(super::ContextError::IncorrectOutputSizeError(
                    rep_with.len(),
                    params_with_name.len(),
                ));
            }

            for i in 0..params_with_name.len() {
                let param_node = &self.nodes[params_with_name[i]];
                let rep_with_node = &self.nodes[rep_with[i]];

                if param_node.shape != rep_with_node.shape || param_node.dtype != rep_with_node.dtype {
                    return Err(super::ContextError::InvalidFuseTargetsError(param_node.dtype, rep_with_node.dtype))
                }
                self.nodes[params_with_name[i]] = self.nodes[rep_with[i]].clone();

                let param_idx = self.parameters.iter().enumerate().find(|(_, node)| node == &&params_with_name[i]);
                if let Some((id, _)) = param_idx {
                    self.parameters.remove(id);
                }

                /*let node_ext = self.dependent_nodes.get(&rep_with[i]).unwrap_or(&vec![]).clone();
                if let Some(node_deps) = self.dependent_nodes.get_mut(&params_with_name[i]) {
                    node_deps.extend(node_ext.iter())
                }*/

                //Add param nodeid to dependent nodes of new node's operation
                match self.nodes[params_with_name[i]].operation.clone() {
                    Operation::Add(a, b)
                        | Operation::Pow(a, b)
                        | Operation::Sub(a, b)
                        | Operation::Mul(a, b)
                        | Operation::MatMul(a, b)
                        | Operation::Div(a, b)
                        | Operation::GreaterThanEq(a, b)
                        | Operation::GreaterThan(a, b)
                        | Operation::Equal(a, b)
                        | Operation::NotEqual(a, b)
                        | Operation::LessThan(a, b)
                        | Operation::LessThanEq(a, b)
                        | Operation::RngUniform(a, b, _)
                        | Operation::RngNormal(a, b, _) => {
                            self.dependent_nodes.entry(a).or_insert_with(Vec::new).push(params_with_name[i]);
                            self.dependent_nodes.entry(b).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::StopGradient(a)
                        | Operation::Neg(a)
                        | Operation::Log(a)
                        | Operation::Exp(a)
                        | Operation::ZerosLike(a)
                        | Operation::OneHot(a)
                        | Operation::TypeCast(a, _)
                        | Operation::Reshape(a)
                        | Operation::Transpose(a, _) => {
                            self.dependent_nodes.entry(a).or_insert_with(Vec::new).push(params_with_name[i]);
                        }
                    Operation::Select {
                        pred,
                        on_false,
                        on_true,
                    } => {
                        self.dependent_nodes.entry(pred).or_insert_with(Vec::new).push(params_with_name[i]);
                        self.dependent_nodes.entry(on_true).or_insert_with(Vec::new).push(params_with_name[i]);
                        self.dependent_nodes.entry(on_false).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::ReduceMax { node, dim: _ } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::ReduceArgmax {
                        node,
                        dim: _,
                    } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::ReduceSum { node, dim: _ } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::ReduceMean { node, dim: _ } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::SliceInDim {
                        node,
                        start: _,
                        stop: _,
                        stride: _,
                        dim: _,
                    } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    Operation::TileInDim { node, n_tiles: _, dim: _ } => {
                        self.dependent_nodes.entry(node).or_insert_with(Vec::new).push(params_with_name[i]);
                    }
                    _ => {} //Constants, parameters, don't need nodeid replacement
                }
            }
        }

        Ok(())
    }

    fn inputs(&self) -> Vec<NodeIdentifier> {
        let mut res = self.parameters.clone();
        res.extend(self.constants.iter());

        res
    }
}
