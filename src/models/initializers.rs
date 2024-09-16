use crate::graph::shape::Shape;

/// `Initializer` represents a function which can initialize the parameters of a network.
///
/// Given a seed, shape, and data type, the `initialize` function should return a random
/// tensor of the given shape and data type.
pub trait Initializer {
    fn initialize(
        &self,
        seed: i64,
        shape: &Shape,
        dtype: xla::ElementType,
    ) -> xla::Result<xla::Literal>;
}

/// `ConstInit` struct as an instance of `Initializer` that fills tensors with a given constant.
pub struct ConstInit {
    pub constant: f32
}

impl Initializer for ConstInit {
    fn initialize(
        &self,
        _: i64,
        shape: &Shape,
        dtype: xla::ElementType,
    ) -> xla::Result<xla::Literal> {
        let const_vec = [self.constant].repeat(shape.size());
        let const_r1 = xla::Literal::vec1(&const_vec);
        let shape_i64 = shape.sizes.iter().map(|x| *x as i64).collect::<Vec<i64>>();
        let const_shaped = xla::Literal::reshape(&const_r1, &shape_i64)?;
        xla::Literal::convert(&const_shaped, dtype.primitive_type())
    }
}
