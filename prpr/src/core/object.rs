use super::{AnimFloat, AnimVector, Color, Matrix, Resource, Vector};
use macroquad::prelude::*;
use nalgebra::Rotation2;

#[derive(Default, Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Object {
    pub alpha: AnimFloat,
    pub scale: AnimVector,
    pub rotation: AnimFloat,
    pub translation: AnimVector,
}

impl Object {
    pub fn is_default(&self) -> bool {
        self.alpha.is_default()
            && self.scale.0.is_default()
            && self.scale.1.is_default()
            && self.rotation.is_default()
            && self.translation.0.is_default()
            && self.translation.1.is_default()
    }

    pub fn set_time(&mut self, time: f32) {
        self.alpha.set_time(time);
        self.scale.0.set_time(time);
        self.scale.1.set_time(time);
        self.rotation.set_time(time);
        self.translation.0.set_time(time);
        self.translation.1.set_time(time);
    }

    pub fn dead(&self) -> bool {
        self.alpha.dead()
            && self.scale.0.dead()
            && self.scale.1.dead()
            && self.rotation.dead()
            && self.translation.0.dead()
            && self.translation.1.dead()
    }

    pub fn now(&self, res: &Resource) -> Matrix {
        self.now_rotation().append_translation(&self.now_translation(res))
    }

    #[inline]
    pub fn now_rotation(&self) -> Matrix {
        Rotation2::new(self.rotation.now().to_radians()).to_homogeneous()
    }

    #[inline]
    pub fn now_translation(&self, res: &Resource) -> Vector {
        let mut tr = self.translation.now();
        tr.y /= res.aspect_ratio;
        tr
    }

    #[inline]
    pub fn now_alpha(&self) -> f32 {
        self.alpha.now_opt().unwrap_or(1.0).max(0.)
    }

    #[inline]
    pub fn now_color(&self) -> Color {
        Color::new(1.0, 1.0, 1.0, self.now_alpha())
    }

    #[inline]
    pub fn now_scale(&self) -> Matrix {
        Matrix::identity().append_nonuniform_scaling(&self.scale.now_with_def(1.0, 1.0))
    }

    pub fn now_scale_fix(&self, ct: Vector) -> Matrix {
        let scale = self.scale.now_with_def(1.0, 1.0);
        Matrix::new_translation(&-ct).append_nonuniform_scaling(&scale).append_translation(&ct)
    }

    pub fn new_rotation_wrt_point(&self, angle: f32, pt: Vector) -> Matrix {
        let rot = Rotation2::new(angle);
        let translation_back = Matrix::new_translation(&pt);
        let translation_to = Matrix::new_translation(&-pt);
        translation_back * rot.to_homogeneous() * translation_to
    }
}

#[derive(Default, Clone)]
pub struct CtrlObject {
    pub alpha: AnimFloat,
    pub size: AnimFloat,
    pub pos: AnimFloat,
    pub y: AnimFloat,
}

impl CtrlObject {
    #[inline]
    fn curve_is_identity(curve: &AnimFloat) -> bool {
        curve.is_default() || curve.constant_value() == Some(1.0)
    }

    #[inline]
    pub(crate) fn is_identity(&self) -> bool {
        Self::curve_is_identity(&self.alpha)
            && Self::curve_is_identity(&self.size)
            && Self::curve_is_identity(&self.pos)
            && Self::curve_is_identity(&self.y)
    }

    #[inline]
    pub(crate) fn y_is_identity(&self) -> bool {
        Self::curve_is_identity(&self.y)
    }

    pub fn set_height(&mut self, height: f32) {
        self.alpha.set_time(height);
        self.size.set_time(height);
        self.pos.set_time(height);
        self.y.set_time(height);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Keyframe;

    #[test]
    fn identity_controls_match_sampling_at_all_heights() {
        let mut ctrl = CtrlObject {
            alpha: AnimFloat::new(vec![Keyframe::new(-2., 1., 2), Keyframe::new(3., 1., 2)]),
            size: AnimFloat::chain(vec![AnimFloat::fixed(0.25), AnimFloat::fixed(0.75)]),
            pos: AnimFloat::fixed(1.),
            y: AnimFloat::default(),
        };
        assert!(ctrl.is_identity());
        for height in [-10., -2., 0., 1., 3., 10.] {
            ctrl.set_height(height);
            for curve in [&ctrl.alpha, &ctrl.size, &ctrl.pos, &ctrl.y] {
                assert_eq!(curve.now_opt().unwrap_or(1.), 1.);
            }
        }
    }

    #[test]
    fn identity_fast_path_rejects_animation_and_near_identity_values() {
        let varying = AnimFloat::new(vec![Keyframe::new(0., 1., 2), Keyframe::new(1., 0.5, 2)]);
        let changed_layer = AnimFloat::chain(vec![AnimFloat::fixed(1.), varying.clone()]);
        for curve in [varying, changed_layer, AnimFloat::fixed(f32::from_bits(1f32.to_bits() + 1))] {
            for field in 0..4 {
                let mut ctrl = CtrlObject::default();
                *match field {
                    0 => &mut ctrl.alpha,
                    1 => &mut ctrl.size,
                    2 => &mut ctrl.pos,
                    _ => &mut ctrl.y,
                } = curve.clone();
                assert!(!ctrl.is_identity());
                assert_eq!(ctrl.y_is_identity(), field != 3);
            }
        }
    }
}
