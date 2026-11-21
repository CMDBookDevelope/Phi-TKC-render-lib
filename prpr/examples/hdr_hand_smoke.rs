//! Run with `cargo run -p prpr --example hdr_hand_smoke` on a Vulkan host.
//! On headless Linux this needs a playback device (an ALSA null PCM works).
use macroquad::{miniquad::conf::RenderingBackend, prelude::*};
use prpr::{
    config::Config,
    core::MSRenderTarget,
    fs::fs_from_file,
    info::{ChartFormat, ChartInfo},
    scene::{GameMode, GameScene, Scene},
    time::TimeManager,
    ui::{FontArc, TextPainter, Ui},
};

fn coverage(target: &MSRenderTarget) -> Vec<(u32, u32)> {
    let (width, height) = target.dimensions();
    let mut bytes = vec![0; (width * height * 8) as usize];
    let gl = unsafe { get_internal_gl() };
    target
        .output()
        .texture
        .raw_miniquad_texture_handle()
        .read_pixels(gl.quad_context, &mut bytes);
    bytes
        .chunks_exact(8)
        .enumerate()
        .filter_map(|(i, pixel)| {
            let alpha = u16::from_ne_bytes([pixel[6], pixel[7]]);
            (alpha > 0x3800).then_some((i as u32 % width, i as u32 / width))
        })
        .collect()
}

fn bounds(points: &[(u32, u32)]) -> (u32, u32, u32, u32) {
    (
        points.iter().map(|p| p.0).min().expect("no visible hand notes"),
        points.iter().map(|p| p.0).max().unwrap(),
        points.iter().map(|p| p.1).min().unwrap(),
        points.iter().map(|p| p.1).max().unwrap(),
    )
}

fn main() {
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let _runtime = runtime.enter();
    let mut conf = Conf {
        headless: true,
        window_width: 224,
        window_height: 144,
        ..Default::default()
    };
    conf.platform.rendering_backend = RenderingBackend::Vulkan;
    macroquad::Window::from_config(conf, async move {
        let assets = concat!(env!("CARGO_MANIFEST_DIR"), "/../assets");
        set_pc_assets_folder(assets);
        let files = tempfile::tempdir().unwrap();
        std::fs::write(files.path().join("song.ogg"), include_bytes!("../../assets/cali.ogg")).unwrap();
        // Two simultaneously hittable notes, on opposite sides and at different heights.
        std::fs::write(
            files.path().join("chart.pec"),
            concat!(
                "0\nbp 0 120\n",
                "cp 0 0 1024 950\ncv 0 0 5.85\nca 0 0 255\nn1 0 2 -512 1 0\n",
                "cp 1 0 1024 280\ncv 1 0 5.85\nca 1 0 255\nn1 1 2 512 1 0\n",
            ),
        )
        .unwrap();
        let info = ChartInfo {
            chart: "chart.pec".into(),
            format: Some(ChartFormat::Pec),
            music: "song.ogg".into(),
            ..Default::default()
        };
        let config = Config {
            hdr_output: true,
            hand_split: true,
            interactive: false,
            bar: false,
            ui_score: false,
            ui_combo: false,
            ui_name: false,
            ui_level: false,
            ui_pb: false,
            ui_pause: false,
            ..Default::default()
        };
        let black = Texture2D::from_rgba8(1, 1, &[0, 0, 0, 255]);
        let mut game =
            GameScene::new(GameMode::View, info, config, fs_from_file(files.path()).unwrap(), None, black.into(), black.into(), None, None)
                .await
                .unwrap();
        let mut painter = TextPainter::new(FontArc::try_from_slice(include_bytes!("../../assets/bold.ttf")).unwrap());
        let output = MSRenderTarget::new_hdr((224, 144), 1);
        let viewport = (24, 13, 176, 112);
        let mut ui = Ui::new(&mut painter, Some(viewport));
        let mut time = TimeManager::manual(Box::new(|| 1.0));
        game.res.update_size(viewport);
        game.res.time = 0.93;
        game.chart.update(&mut game.res);
        game.render(&mut time, &mut ui, Some(output.input())).unwrap();
        let hand_pixels = game.res.hdr_hand_targets.each_ref().map(|target| coverage(target.as_ref().unwrap()));
        let split_bounds = hand_pixels.each_ref().map(|points| bounds(points));
        assert_eq!(split_bounds[0].0 < split_bounds[1].0, true, "{split_bounds:?}");
        assert!(split_bounds[0].2.abs_diff(split_bounds[1].2) > 30, "asymmetric notes: {split_bounds:?}");
        let split_lines = coverage(game.res.hdr_line_target.as_ref().unwrap());

        game.res.config.hand_split = false;
        game.chart.update(&mut game.res);
        game.render(&mut time, &mut ui, Some(output.input())).unwrap();
        let plain = coverage(game.res.chart_target.as_ref().unwrap());
        let plain_bounds = [
            bounds(&plain.iter().copied().filter(|p| p.0 < 88).collect::<Vec<_>>()),
            bounds(&plain.iter().copied().filter(|p| p.0 >= 88).collect::<Vec<_>>()),
        ];
        assert_eq!(split_bounds, plain_bounds, "hand layers must follow the chart projection");
        assert_eq!(split_lines, coverage(game.res.hdr_line_target.as_ref().unwrap()));
        println!("HDR AI hand notes match unseparated notes and preserve line coverage in viewport {viewport:?}: {split_bounds:?}");
        std::process::exit(0);
    });
}
