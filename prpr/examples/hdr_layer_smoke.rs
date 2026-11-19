use macroquad::prelude::*;
use macroquad::miniquad::conf::RenderingBackend;
use prpr::{config::HdrLayerSettings, core::MSRenderTarget};

fn main() {
    let mut conf = Conf { headless: true, window_width: 64, window_height: 64, ..Default::default() };
    conf.platform.rendering_backend = RenderingBackend::Vulkan;
    macroquad::Window::from_config(conf, async move {
        let mut image = MSRenderTarget::new_hdr((4, 4), 1);
        set_camera(&Camera2D { zoom: vec2(1., -1.), render_target: Some(image.input()), ..Default::default() });
        clear_background(BLANK);
        draw_rectangle(-1., -1., 2., 1., Color::new(1., 1., 1., 0.5));
        let settings = HdrLayerSettings { mode: "mapped".into(), ..Default::default() };
        image.map_hdr(&settings);
        let mut bytes = [0u8;128];
        let gl = unsafe { get_internal_gl() };
        image.output().texture.raw_miniquad_texture_handle().read_pixels(gl.quad_context, &mut bytes);
        let first = u16::from_ne_bytes(bytes[0..2].try_into().unwrap());
        let last = u16::from_ne_bytes(bytes[120..122].try_into().unwrap());
        let alpha = u16::from_ne_bytes(bytes[6..8].try_into().unwrap());
        let last_alpha = u16::from_ne_bytes(bytes[126..128].try_into().unwrap());
        let pixels = [(first,alpha),(last,last_alpha)];
        assert!(pixels.iter().any(|&(red, alpha)| red.abs_diff(0x4900) <= 2 && alpha.abs_diff(0x3800) <= 16), "{pixels:x?}");
        assert!(pixels.contains(&(0,0)), "{pixels:x?}");
        println!("Mapped reference white to 1000 nits with fractional alpha: {pixels:x?}");
        let size = (128, 128);
        let base = MSRenderTarget::new_hdr(size, 1);
        let mut layer = MSRenderTarget::new_hdr(size, 1);
        set_camera(&Camera2D { zoom: vec2(1., 1.), render_target: Some(base.input()), ..Default::default() });
        clear_background(WHITE);
        let zone = prpr::core::Zone {
            center: prpr::core::Vector::zeros(),
            half: prpr::core::Vector::new(0.30, 0.30),
            angle: 0.,
            invert: false,
            active: false,
        };
        set_camera(&Camera2D { zoom: vec2(1., 1.), render_target: Some(layer.input()), ..Default::default() });
        clear_background(BLANK);
        prpr::core::draw_block_effect(&[zone], 1., (128, 128, 128, 128), base.output().texture, 1., (-1., -1., 2., 2.));
        layer.map_hdr_linear_bt2020(&settings);
        let mut bytes = vec![0; 128 * 128 * 8];
        let read = |target: &MSRenderTarget, bytes: &mut [u8]| {
            let gl = unsafe { get_internal_gl() };
            target.output().texture.raw_miniquad_texture_handle().read_pixels(gl.quad_context, bytes);
        };
        read(&layer, &mut bytes);
        let center = (64 * 128 + 64) * 8;
        assert_eq!(&bytes[..8], &[0; 8]);
        assert_ne!(&bytes[center..center+8], &[0; 8]);

        set_camera(&Camera2D { zoom: vec2(1., 1.), render_target: Some(layer.input()), ..Default::default() });
        clear_background(WHITE);
        draw_rectangle(-0.25, -0.25, 0.5, 0.5, Color::new(0.5, 0.5, 0.5, 1.));
        layer.map_hdr_changed_pixels(&settings, base.output().texture);
        read(&layer, &mut bytes);
        assert_eq!(&bytes[..8], &[0; 8]);
        assert_ne!(&bytes[center..center+8], &[0; 8]);
        println!("Mapped only procedural coverage and changed RPE-style pixels");
        std::process::exit(0);
    });
}
