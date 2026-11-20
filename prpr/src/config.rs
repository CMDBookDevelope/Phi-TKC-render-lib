use bitflags::bitflags;
use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

unsafe impl Send for Config {}
unsafe impl Sync for Config {}

pub static TIPS: Lazy<Vec<String>> = Lazy::new(|| {
    include_str!("tips.txt")
        .split('\n')
        //.map(str::to_owned)
        .map(|s| format!("Tip: {}", s))
        .collect()
});

bitflags! {
    #[derive(Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq, Debug)]
    #[serde(transparent)]
    pub struct Mods: i32 {
        const AUTOPLAY = 1;
        const FLIP_X = 2;
        const FADE_OUT = 4;
    }
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ChallengeModeColor {
    White,
    Green,
    Blue,
    Red,
    Golden,
    Rainbow,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct HdrLayerSettings {
    pub mode: String,
    pub peak_nits: f32,
    pub knee: f32,
    pub power: f32,
    pub shadow_gain: f32,
    pub chroma_scale: f32,
    pub gamut_clip: String,
}

impl Default for HdrLayerSettings {
    fn default() -> Self {
        Self {
            mode: "sdr".into(),
            peak_nits: 1000.,
            knee: 0.70,
            power: 1.70,
            shadow_gain: 1.0,
            chroma_scale: 1.0,
            gamut_clip: "hard".into(),
        }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct HdrSettings {
    pub block_area: HdrLayerSettings,
    pub note: HdrLayerSettings,
    pub background: HdrLayerSettings,
    pub rhythm: RhythmHdrSettings,
    pub hand_indicator: HandIndicatorHdrSettings,
    pub hit_effect: HdrLayerSettings,
    pub ui: HashMap<String, HdrLayerSettings>,
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct RhythmHdrSettings {
    pub metronome: HdrLayerSettings,
    pub note: HdrLayerSettings,
    pub value: HdrLayerSettings,
}

impl Default for RhythmHdrSettings {
    fn default() -> Self {
        let layer = HdrLayerSettings::default();
        Self { metronome: layer.clone(), note: layer.clone(), value: layer }
    }
}

#[derive(Clone, Deserialize, Serialize, Debug)]
#[serde(rename_all = "camelCase", default)]
pub struct HandIndicatorHdrSettings {
    pub left: HdrLayerSettings,
    pub right: HdrLayerSettings,
}

impl Default for HandIndicatorHdrSettings {
    fn default() -> Self {
        let layer = HdrLayerSettings::default();
        Self { left: layer.clone(), right: layer }
    }
}

impl Default for HdrSettings {
    fn default() -> Self {
        let default_layer = HdrLayerSettings::default();
        let ui = [
            "score",
            "combo",
            "name",
            "level",
            "line",
            "pb",
            "pause",
            "progressText",
            "timeText",
            "bar",
            "scoreAim",
            "watermark",
        ]
        .into_iter()
        .map(|key| (key.to_owned(), default_layer.clone()))
        .collect();
        Self {
            block_area: default_layer.clone(),
            note: default_layer.clone(),
            background: default_layer.clone(),
            rhythm: RhythmHdrSettings::default(),
            hand_indicator: HandIndicatorHdrSettings::default(),
            hit_effect: default_layer,
            ui,
        }
    }
}

/// Map one SDR channel into linear light normalized to 100 nits.
/// `mode=sdr` preserves reference-white luminance, while mapped/custom modes
/// apply a bounded highlight shoulder without changing black.
pub fn hdr_map_channel(value: f32, settings: &HdrLayerSettings) -> f32 {
    let value = value.max(0.0);
    if settings.mode == "sdr" {
        return value;
    }
    let peak = (settings.peak_nits.max(100.0) / 100.0).max(1.0);
    let gain = settings.shadow_gain.max(0.0);
    let x = value * gain;
    let knee = settings.knee.clamp(0.01, 0.99);
    if settings.mode == "mapped" {
        return x * peak;
    }
    let power = settings.power.clamp(0.1, 8.0);
    if x <= knee {
        return x;
    }
    let source_span = (1.0 - knee).max(f32::EPSILON);
    let output_span = (peak - knee).max(0.0);
    let t = ((x - knee) / source_span).clamp(0.0, 1.0);
    let shaped = 1.0 - (1.0 - t).powf(power);
    let mut mapped = knee + shaped * output_span;
    if settings.gamut_clip == "soft" {
        mapped = peak * (mapped / peak).max(0.0).tanh() / peak.tanh();
    }
    mapped.min(peak)
}

/// Convert display-referred sRGB bytes to linear BT.2020 light with 1.0 = 100 nits.
pub fn hdr_map_srgb(rgb: [u8; 3], settings: &HdrLayerSettings) -> [f32; 3] {
    let linear = rgb.map(|channel| {
        let v = channel as f32 / 255.0;
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    });
    let mapped = linear.map(|channel| hdr_map_channel(channel, settings));
    let mut bt2020 = [
        0.6274 * mapped[0] + 0.3293 * mapped[1] + 0.0433 * mapped[2],
        0.0691 * mapped[0] + 0.9195 * mapped[1] + 0.0114 * mapped[2],
        0.0164 * mapped[0] + 0.0880 * mapped[1] + 0.8956 * mapped[2],
    ];
    let luma = 0.2627 * bt2020[0] + 0.6780 * bt2020[1] + 0.0593 * bt2020[2];
    let chroma = settings.chroma_scale.max(0.0);
    for channel in &mut bt2020 {
        *channel = luma + (*channel - luma) * chroma;
        if settings.gamut_clip == "soft" {
            let peak = (settings.peak_nits.max(100.0) / 100.0).max(1.0);
            *channel = peak * (*channel / peak).tanh() / peak.tanh();
        } else {
            *channel = channel.max(0.0);
        }
    }
    bt2020
}

#[cfg(test)]
mod hdr_tests {
    use super::*;

    #[test]
    fn sdr_layer_keeps_reference_white_at_100_nits() {
        let settings = HdrLayerSettings::default();
        assert_eq!(hdr_map_channel(1.0, &settings), 1.0);
        assert!(hdr_map_srgb([255; 3], &settings).iter().all(|channel| (*channel - 1.0).abs() < 0.002));
    }

    #[test]
    fn mapped_layer_scales_linear_light_by_peak() {
        let settings = HdrLayerSettings {
            mode: "mapped".into(),
            ..Default::default()
        };
        assert_eq!(hdr_map_channel(0.5, &settings), 5.0);
    }

    #[test]
    fn hdr_nested_sublayers_default_independently_for_legacy_and_partial_configs() {
        let legacy: Config = serde_json::from_str("{}").unwrap();
        assert_eq!(legacy.hdr_settings.rhythm.metronome.mode, "sdr");
        assert_eq!(legacy.hdr_settings.rhythm.note.mode, "sdr");
        assert_eq!(legacy.hdr_settings.hand_indicator.left.mode, "sdr");
        assert_eq!(legacy.hdr_settings.hit_effect.mode, "sdr");
        assert_eq!(legacy.hand_left_color, "#3380FF");
        assert_eq!(legacy.hand_right_color, "#FF99B3");
        assert_eq!(legacy.hand_left_opacity, 1.0);
        assert_eq!(legacy.hand_right_opacity, 1.0);
        assert!(legacy.hit_effect_enabled);

        let partial: Config = serde_json::from_str(
            r#"{"hdrSettings":{"rhythm":{"note":{"mode":"mapped","peakNits":2500}},"handIndicator":{"right":{"mode":"custom"}}}}"#,
        ).unwrap();
        assert_eq!(partial.hdr_settings.rhythm.note.mode, "mapped");
        assert_eq!(partial.hdr_settings.rhythm.note.peak_nits, 2500.0);
        assert_eq!(partial.hdr_settings.rhythm.metronome.mode, "sdr");
        assert_eq!(partial.hdr_settings.rhythm.value.mode, "sdr");
        assert_eq!(partial.hdr_settings.hand_indicator.left.mode, "sdr");
        assert_eq!(partial.hdr_settings.hand_indicator.right.mode, "custom");
        assert_eq!(partial.hdr_settings.hit_effect.mode, "sdr");
    }

    #[test]
    fn custom_hand_tint_values_round_trip_and_hit_effect_toggle_defaults_on() {
        let config: Config = serde_json::from_str(
            r##"{"handLeftColor":"#123abc","handRightOpacity":0.35,"hitEffectEnabled":false}"##,
        ).unwrap();
        assert_eq!(config.hand_left_color, "#123abc");
        assert_eq!(config.hand_right_opacity, 0.35);
        assert!(!config.hit_effect_enabled);
        assert_eq!(config.hand_left_opacity, 1.0);
    }

    #[test]
    fn custom_layer_is_bounded_and_monotonic() {
        let settings = HdrLayerSettings {
            mode: "custom".into(),
            ..Default::default()
        };
        assert_eq!(hdr_map_channel(0.5, &settings), 0.5);
        assert_eq!(hdr_map_channel(1.0, &settings), 10.0);
        let values = [0.0, 0.5, 1.0, 2.0, 20.0].map(|v| hdr_map_channel(v, &settings));
        assert!(values.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(values.iter().all(|v| *v <= 10.0));
    }
}

#[derive(Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum BackgroundVideoPlayback {
    #[default]
    Normal,
    Repeat,
    Stretch,
}

#[derive(Clone, Deserialize, Serialize)]
#[serde(default)]
#[serde(rename_all = "camelCase")]
pub struct Config {
    #[serde(rename = "adjust_time_new")]
    pub adjust_time: bool,
    pub aggressive: bool,
    pub audio_buffer_size: Option<u32>,
    pub challenge_color: ChallengeModeColor,
    pub challenge_rank: u32,
    pub chart_debug: bool,
    pub chart_ratio: f32,
    pub buffer_size: f32,
    pub combo: String,
    pub flid_x: bool,
    pub disable_effect: bool,
    pub double_click_to_pause: bool,
    pub double_hint: bool,
    pub fxaa: bool,
    pub interactive: bool,
    pub note_scale: f32,
    pub mods: Mods,
    pub mp_enabled: bool,
    pub mp_address: String,
    pub offline_mode: bool,
    pub offset: f32,
    pub particle: bool,
    pub player_name: String,
    pub player_rks: f32,
    pub res_pack_path: Option<String>,
    pub sample_count: u32,
    pub show_acc: bool,
    pub speed: f32,
    pub disable_loading: bool,
    pub target_audio: u32,
    // test
    pub show_progress_text: bool,
    pub show_time_text: bool,
    pub watermark: String,
    pub background: bool,
    /// Background source selected by the renderer (`chart`, `solid`, `video`, `none`).
    pub background_mode: String,
    /// Linearized RGBA color for a solid background.
    pub background_color: [f32; 4],
    /// Optional local video path used as the export/playback background.
    pub background_video: Option<String>,
    pub background_video_playback: BackgroundVideoPlayback,
    #[serde(skip)]
    pub background_video_interval: Option<(f64, f64)>,
    #[serde(skip)]
    pub transparent_background: bool,
    /// Render into linear-light HDR targets when the host has negotiated an HDR output.
    #[serde(skip)]
    pub hdr_output: bool,
    /// Transfer function requested for gameplay and preview presentation.
    /// Unsupported HDR displays keep the SDR presentation path.
    pub color_space: String,
    /// Per-layer mapping controls used when HDR presentation is available.
    pub hdr_settings: HdrSettings,
    /// Hex tint colors used for left/right hand-split notes.
    pub hand_left_color: String,
    pub hand_right_color: String,
    /// Hand tint strength. 1.0 preserves the historical full tint.
    pub hand_left_opacity: f32,
    pub hand_right_opacity: f32,
    /// Show judgement hit effects (bad-note flashes and hit particles).
    pub hit_effect_enabled: bool,
    //test
    pub touch_debug: bool,
    pub volume_music: f32,
    pub volume_sfx: f32,
    pub volume_bgm: f32,
    pub hand_split: bool,
    pub hand_ai_epochs: u32,
    pub hand_ai_train_budget_ms: u32,
    pub note_speed_factor: f32,
    pub bar: bool,
    pub combo_anim: bool,
    pub emitting: bool,
    pub score_aim: bool,
    /// Accepted for compatibility with configs written by other forks (which
    /// used it to pick between a GPU shader and a flat fallback). This branch
    /// has a single 噪域 renderer, so the value is ignored.
    pub block_area_simple: bool,
    /// Draw the HUD above 噪域 (`blockAreaList`) instead of letting the effect
    /// cover it. Off by default: the reference renderer keeps the HUD under the
    /// effect.
    pub block_area_ui: bool,
    /// Append the rhythm analysis strip to exported video.
    pub rhythm_analysis: bool,
    /// Infer musical tempo from decoded audio only when the chart has no BPM map.
    pub rhythm_auto_bpm: bool,
    pub rhythm_tap: bool,
    pub rhythm_drag: bool,
    pub rhythm_hold: bool,
    pub rhythm_flick: bool,
    /// Visual timeline speed only; musical tempo and hit times are unchanged.
    pub rhythm_speed: f32,
    /// Exact #RRGGBB accents, or automatic colors from the chart illustration.
    pub rhythm_colors: Option<[String; 3]>,

    // for compatibility
    pub autoplay: Option<bool>,

    pub ui_score: bool,
    pub ui_combo: bool,
    pub ui_name: bool,
    pub ui_level: bool,
    pub ui_line: bool,
    pub ui_pb: bool,
    pub ui_pause: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            adjust_time: false,
            aggressive: false,
            audio_buffer_size: None,
            challenge_color: ChallengeModeColor::Golden,
            challenge_rank: 45,
            chart_debug: false,
            chart_ratio: 1.0,
            buffer_size: 256.0,
            combo: "COMBO".to_string(),
            flid_x: false,
            disable_effect: false,
            double_click_to_pause: true,
            double_hint: true,
            fxaa: false,
            interactive: true,
            mods: Mods::default(),
            mp_address: "mp2.phira.cn:12345".to_owned(),
            mp_enabled: false,
            note_scale: 1.0,
            offline_mode: false,
            offset: 0.,
            particle: true,
            player_name: "Link".to_string(),
            player_rks: 15.,
            res_pack_path: None,
            sample_count: 1,
            show_acc: false,
            speed: 1.,
            touch_debug: false,
            volume_music: 1.,
            volume_sfx: 1.,
            volume_bgm: 1.,
            watermark: "".to_string(),
            background: false,
            background_mode: "chart".to_owned(),
            background_color: [0., 0., 0., 1.],
            background_video: None,
            background_video_playback: BackgroundVideoPlayback::Normal,
            background_video_interval: None,
            transparent_background: false,
            hdr_output: false,
            color_space: "rec709".to_owned(),
            hdr_settings: HdrSettings::default(),
            hand_left_color: "#3380FF".to_owned(),
            hand_right_color: "#FF99B3".to_owned(),
            hand_left_opacity: 1.0,
            hand_right_opacity: 1.0,
            hit_effect_enabled: true,
            disable_loading: false,
            target_audio: 96000,

            autoplay: None,
            show_progress_text: false,
            show_time_text: false,
            hand_split: false,
            hand_ai_epochs: 4,
            hand_ai_train_budget_ms: 50,
            note_speed_factor: 1.0,
            bar: true,
            combo_anim: false,
            emitting: false,
            score_aim: false,
            block_area_simple: true,
            block_area_ui: false,
            rhythm_analysis: false,
            rhythm_auto_bpm: true,
            rhythm_tap: true,
            rhythm_drag: false,
            rhythm_hold: true,
            rhythm_flick: false,
            rhythm_speed: 1.,
            rhythm_colors: None,

            ui_score: true,
            ui_combo: true,
            ui_name: true,
            ui_level: true,
            ui_line: true,
            ui_pb: true,
            ui_pause: true,
        }
    }
}

impl Config {
    pub fn init(&mut self) {
        if let Some(flag) = self.autoplay {
            self.mods.set(Mods::AUTOPLAY, flag);
        }
    }

    #[inline]
    pub fn has_mod(&self, m: Mods) -> bool {
        self.mods.contains(m)
    }

    #[inline]
    pub fn autoplay(&self) -> bool {
        self.has_mod(Mods::AUTOPLAY)
    }

    #[inline]
    pub fn flip_x(&self) -> bool {
        self.flid_x || self.has_mod(Mods::FLIP_X)
    }
}
