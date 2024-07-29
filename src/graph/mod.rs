pub mod autodiff;
pub(crate) mod callsite;
pub mod compile;
pub mod merge;
pub mod constant;
pub mod consteval;
pub mod context;
pub mod dtypes;
pub mod logic;
pub mod math;
pub mod node;
pub mod operation;
pub mod parameter;
pub mod shape;
mod subterm;
mod tests;
//mod tests_cpu;

use callsite::{callsite, Callsite};
pub use compile::CompileError;
pub use constant::ConstantBinding;
pub use context::{Context, ContextError, Result};
pub use node::{Node, NodeIdentifier};
pub use operation::Operation;
pub use shape::{Shape, ShapeConversionError};
