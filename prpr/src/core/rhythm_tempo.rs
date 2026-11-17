//! Audio-only tempo fallback for charts which have no authored musical map.
//! The DSP is deliberately std-only so desktop and iOS use the same estimator.

use std::f64::consts::TAU;

const RATE: usize = 8000;
const HOP: usize = 32;
const SPECTRUM: usize = 256;
const WINDOW: f64 = 12.;
const MIN_BPM: usize = 40;
const MAX_BPM: usize = 240;
const MIN_SEGMENT: f64 = 6.;

#[derive(Clone, Debug)]
pub struct TempoPoint {
    /// Audio time in seconds; multiplying BPM never changes this timestamp.
    pub time: f32,
    pub bpm: f32,
}

#[derive(Clone, Debug)]
pub struct TempoEstimate {
    pub points: Vec<TempoPoint>,
    pub first_beat_time: f32,
    pub scale: f32,
}

struct Fft {
    size: usize,
    reverse: Vec<usize>,
    roots: Vec<(f64, f64)>,
}

impl Fft {
    fn new(size: usize) -> Self {
        let bits = size.trailing_zeros();
        Self {
            size,
            reverse: (0..size).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect(),
            roots: (0..size / 2)
                .map(|i| {
                    let angle = -TAU * i as f64 / size as f64;
                    (angle.cos(), angle.sin())
                })
                .collect(),
        }
    }

    fn transform(&self, real: &mut [f64], imag: &mut [f64], inverse: bool) {
        for (i, &j) in self.reverse.iter().enumerate() {
            if i < j {
                real.swap(i, j);
                imag.swap(i, j);
            }
        }
        let mut width = 2;
        while width <= self.size {
            let half = width / 2;
            for base in (0..self.size).step_by(width) {
                for j in 0..half {
                    let (wr, mut wi) = self.roots[j * self.size / width];
                    if inverse {
                        wi = -wi;
                    }
                    let index = base + j + half;
                    let tr = wr * real[index] - wi * imag[index];
                    let ti = wr * imag[index] + wi * real[index];
                    real[index] = real[base + j] - tr;
                    imag[index] = imag[base + j] - ti;
                    real[base + j] += tr;
                    imag[base + j] += ti;
                }
            }
            width *= 2;
        }
        if inverse {
            for (r, i) in real.iter_mut().zip(imag) {
                *r /= self.size as f64;
                *i /= self.size as f64;
            }
        }
    }
}

struct Onsets {
    values: Vec<f64>,
}

impl Onsets {
    fn time(&self, index: usize) -> f64 {
        // A trailing window timestamps the detected attack, rather than
        // assigning flux to the earlier center of its analysis window.
        (index * HOP + SPECTRUM) as f64 / RATE as f64
    }

    fn at(&self, time: f64) -> f64 {
        let position = (time * RATE as f64 - SPECTRUM as f64) / HOP as f64;
        if position < 0. || position >= (self.values.len() - 1) as f64 {
            return 0.;
        }
        let index = position as usize;
        let fraction = position - index as f64;
        self.values[index] * (1. - fraction) + self.values[index + 1] * fraction
    }
}

fn onsets(sample_count: usize, sample_rate: u32, mut sample: impl FnMut(usize) -> f32) -> Option<Onsets> {
    if sample_rate == 0 || sample_count < sample_rate as usize * 4 {
        return None;
    }
    let count = ((sample_count as u64 * RATE as u64) / sample_rate as u64) as usize;
    let step = sample_rate as f64 / RATE as f64;
    let mut mono = Vec::with_capacity(count);
    if sample_rate as usize <= RATE {
        for i in 0..count {
            let position = i as f64 * step;
            let left = (position as usize).min(sample_count - 1);
            let fraction = position - left as f64;
            let a = sample(left);
            let b = sample((left + 1).min(sample_count - 1));
            let finite = |value: f32| if value.is_finite() { value as f64 } else { 0. };
            mono.push(finite(a) * (1. - fraction) + finite(b) * fraction);
        }
    } else {
        // High-frequency percussion must be removed before decimation: a box
        // average aliases it into false low-frequency onset periodicities.
        // A small polyphase windowed-sinc filter avoids trig in the audio loop.
        const PHASES: usize = 128;
        let radius = (step * 8.).ceil() as isize;
        let cutoff = 0.47 / step;
        let kernels: Vec<Vec<f64>> = (0..PHASES)
            .map(|phase| {
                let fraction = phase as f64 / PHASES as f64;
                let mut weights: Vec<_> = (-radius + 1..=radius)
                    .map(|tap| {
                        let distance = tap as f64 - fraction;
                        let angle = TAU * cutoff * distance;
                        let sinc = if distance.abs() < 1e-12 {
                            2. * cutoff
                        } else {
                            angle.sin() / (std::f64::consts::PI * distance)
                        };
                        sinc * (0.5 + 0.5 * (std::f64::consts::PI * distance / radius as f64).cos())
                    })
                    .collect();
                let sum = weights.iter().sum::<f64>();
                for weight in &mut weights {
                    *weight /= sum;
                }
                weights
            })
            .collect();
        for i in 0..count {
            let position = i as f64 * step;
            let center = position as isize;
            let phase = (((position - center as f64) * PHASES as f64).round() as usize).min(PHASES - 1);
            let mut sum = 0.;
            for (index, &weight) in kernels[phase].iter().enumerate() {
                let source = (center - radius + 1 + index as isize).clamp(0, sample_count as isize - 1) as usize;
                let value = sample(source);
                if value.is_finite() {
                    sum += value as f64 * weight;
                }
            }
            mono.push(sum);
        }
    }
    if mono.iter().all(|value| value.abs() < 1e-8) {
        return None;
    }
    let fft = Fft::new(SPECTRUM);
    let window: Vec<_> = (0..SPECTRUM)
        .map(|i| 0.5 - 0.5 * (TAU * i as f64 / (SPECTRUM - 1) as f64).cos())
        .collect();
    let mut real = vec![0.; SPECTRUM];
    let mut imag = vec![0.; SPECTRUM];
    let mut previous = vec![0.; SPECTRUM / 2 + 1];
    let mut values = Vec::with_capacity(count / HOP);
    for start in (0..count - SPECTRUM + 1).step_by(HOP) {
        for i in 0..SPECTRUM {
            real[i] = mono[start + i] * window[i];
            imag[i] = 0.;
        }
        fft.transform(&mut real, &mut imag, false);
        let mut flux = 0.;
        for bin in 2..=112 {
            let magnitude = (real[bin] * real[bin] + imag[bin] * imag[bin]).sqrt();
            let current = (1. + 100. * magnitude).ln();
            flux += (current - previous[bin]).max(0.);
            previous[bin] = current;
        }
        values.push(flux / 111.);
    }
    if values.iter().copied().fold(0., f64::max) < 1e-6 {
        return None;
    }
    Some(Onsets { values })
}

struct Observation {
    time: f64,
    scores: Vec<f64>,
    confidence: f64,
}

fn observations(env: &Onsets) -> Vec<Observation> {
    let dt = HOP as f64 / RATE as f64;
    let duration = env.time(env.values.len() - 1);
    let window = WINDOW.min(duration);
    let count = ((window / dt) as usize).min(env.values.len());
    let fft = Fft::new((count * 2).next_power_of_two());
    let mut real = vec![0.; fft.size];
    let mut imag = vec![0.; fft.size];
    let mut rows = Vec::new();
    let centers: Vec<_> = if duration < WINDOW + 1. {
        vec![duration / 2.]
    } else {
        (0..=(duration - WINDOW) as usize).map(|i| WINDOW / 2. + i as f64).collect()
    };
    for center in centers {
        let begin = (((center - window / 2.) / dt).max(0.) as usize).min(env.values.len() - count);
        let values = &env.values[begin..begin + count];
        let mean = values.iter().sum::<f64>() / count as f64;
        real.fill(0.);
        imag.fill(0.);
        let mut energy = vec![0.; count + 1];
        for (i, &value) in values.iter().enumerate() {
            real[i] = value - mean;
            energy[i + 1] = energy[i] + real[i] * real[i];
        }
        fft.transform(&mut real, &mut imag, false);
        for i in 0..fft.size {
            real[i] = real[i] * real[i] + imag[i] * imag[i];
            imag[i] = 0.;
        }
        fft.transform(&mut real, &mut imag, true);
        let max_lag = ((60. / MIN_BPM as f64 / dt * 4.).ceil() as usize + 1).min(count - 1);
        let mut correlation = vec![0.; max_lag + 1];
        for lag in 0..=max_lag {
            let norm = (energy[count - lag] * (energy[count] - energy[lag])).sqrt();
            correlation[lag] = real[lag] / norm.max(1e-12);
        }
        let mut scores = Vec::with_capacity(MAX_BPM - MIN_BPM + 1);
        for bpm in MIN_BPM..=MAX_BPM {
            let mut score = 0.;
            for harmonic in 1..=4 {
                let lag = 60. / bpm as f64 / dt * harmonic as f64;
                let index = lag as usize;
                if index + 1 < correlation.len() {
                    let fraction = lag - index as f64;
                    score += (correlation[index] * (1. - fraction) + correlation[index + 1] * fraction) / harmonic as f64;
                }
            }
            scores.push(score / (1. + 0.5 + 1. / 3. + 0.25));
        }
        // Fractional-lag interpolation and alternating drum accents can make
        // an integer subharmonic stronger than the primitive pulse. Retain
        // the faster pulse when it has substantial periodic support too.
        let raw = scores.clone();
        for bpm in MIN_BPM..=MAX_BPM / 2 {
            for multiple in 2..=MAX_BPM / bpm {
                if raw[bpm * multiple - MIN_BPM] >= (raw[bpm - MIN_BPM] * 0.45).max(0.02) {
                    scores[bpm - MIN_BPM] *= 0.2;
                    break;
                }
            }
        }
        let confidence = scores.iter().copied().fold(0., f64::max);
        // Compare relative evidence across quiet intros and loud later parts.
        // The floor limits amplification of unstructured, near-zero scores.
        let strength = confidence.max(0.05);
        for score in &mut scores {
            *score /= strength;
        }
        rows.push(Observation {
            time: center,
            scores,
            confidence,
        });
    }
    rows
}

fn stable_path(rows: &[Observation]) -> Vec<usize> {
    let count = MAX_BPM - MIN_BPM + 1;
    let mut previous = rows[0].scores.clone();
    let mut links = vec![vec![0usize; count]; rows.len()];
    for (row_index, row) in rows.iter().enumerate().skip(1) {
        let mut next = vec![f64::NEG_INFINITY; count];
        for i in 0..count {
            for j in 0..count {
                let penalty = if i == j {
                    0.
                } else {
                    let ratio = ((i + MIN_BPM) as f64 / (j + MIN_BPM) as f64).max((j + MIN_BPM) as f64 / (i + MIN_BPM) as f64);
                    // Half/double-time drum patterns need more evidence than
                    // an ordinary change, but sustained true halving survives.
                    let metrical = if ratio >= 1.4 && ((ratio - ratio.round()).abs() < 0.02 || (ratio - 1.5).abs() < 0.02) {
                        2.
                    } else {
                        0.
                    };
                    0.20 + 0.15 * ratio.ln() + metrical
                };
                let value = previous[j] - penalty + row.scores[i];
                if value > next[i] {
                    next[i] = value;
                    links[row_index][i] = j;
                }
            }
        }
        previous = next;
    }
    let mut index = (0..count).max_by(|&a, &b| previous[a].total_cmp(&previous[b])).unwrap();
    let mut path = vec![0; rows.len()];
    for row in (0..rows.len()).rev() {
        path[row] = index + MIN_BPM;
        index = links[row][index];
    }
    // Short isolated rhythms must not become bogus tempo changes.
    loop {
        let mut changed = false;
        let mut start = 0;
        while start < path.len() {
            let mut end = start + 1;
            while end < path.len() && path[end] == path[start] {
                end += 1;
            }
            if ((end - start) as f64) < MIN_SEGMENT && !(start == 0 && end == path.len()) {
                let mut choices = Vec::new();
                if start > 0 {
                    choices.push(path[start - 1]);
                }
                if end < path.len() {
                    choices.push(path[end]);
                }
                let best = choices
                    .into_iter()
                    .max_by(|&a, &b| {
                        let score = |bpm: usize| rows[start..end].iter().map(|row| row.scores[bpm - MIN_BPM]).sum::<f64>();
                        score(a).total_cmp(&score(b))
                    })
                    .unwrap();
                path[start..end].fill(best);
                changed = true;
                break;
            }
            start = end;
        }
        if !changed {
            break;
        }
    }
    path
}

fn grid_score(env: &Onsets, start: f64, end: f64, bpm: f64, phase: f64) -> f64 {
    let period = 60. / bpm;
    let mut time = phase + ((start - phase) / period).ceil() * period;
    let mut sum = 0.;
    let mut count = 0.;
    while time < end {
        sum += env.at(time);
        count += 1.;
        time += period;
    }
    if count > 0. {
        sum / count
    } else {
        0.
    }
}

fn phase(env: &Onsets, start: f64, end: f64, bpm: f64) -> f64 {
    let period = 60. / bpm;
    let mut best = (f64::NEG_INFINITY, 0.);
    let mut offset = 0.;
    while offset < period {
        let score = grid_score(env, start, end, bpm, offset);
        if score > best.0 {
            best = (score, offset);
        }
        offset += 0.002;
    }
    best.1
}

fn grid_likelihood(env: &Onsets, start: f64, end: f64, bpm: f64, phase: f64, floor: f64) -> f64 {
    let period = 60. / bpm;
    let mut time = phase + ((start - phase) / period).ceil() * period;
    let mut score = 0.;
    while time < end {
        score += env.at(time) - floor;
        time += period;
    }
    score
}

fn boundary(env: &Onsets, rough: f64, old: f64, new: f64, old_phase: f64, new_phase: f64, lower: f64, upper: f64) -> f64 {
    let start = (rough - WINDOW / 2.).max(lower);
    let end = (rough + WINDOW / 2.).min(upper);
    let period = 60. / old;
    let new_period = 60. / new;
    let floor = 0.5 * grid_score(env, start, rough, old, old_phase).max(grid_score(env, rough, end, new, new_phase));
    let mut candidate = old_phase + ((start - old_phase) / period).ceil() * period;
    let mut best = (f64::NEG_INFINITY, rough);
    while candidate < end {
        let difference = ((candidate - new_phase + new_period / 2.).rem_euclid(new_period) - new_period / 2.).abs();
        if difference <= 0.04 {
            let left = grid_likelihood(env, start, candidate, old, old_phase, floor);
            let right = grid_likelihood(env, candidate, end, new, new_phase, floor);
            let score = left + right - 0.002 * (candidate - rough).abs();
            if score > best.0 {
                best = (score, candidate);
            }
        }
        candidate += period;
    }
    best.1
}

/// Analyze decoded mono samples only when the caller has no authored BPM map.
/// Returns None for disabled/authoritative/invalid/silent input.
pub fn infer_if_missing(
    enabled: bool,
    authored: bool,
    sample_count: usize,
    sample_rate: u32,
    sample: impl FnMut(usize) -> f32,
    chart_first_bpm: f32,
) -> Option<TempoEstimate> {
    if !enabled || authored {
        return None;
    }
    let env = onsets(sample_count, sample_rate, sample)?;
    let rows = observations(&env);
    if rows.is_empty() || rows.iter().map(|row| row.confidence).fold(0., f64::max) < 0.16 {
        return None;
    }
    let chart_bpm = if chart_first_bpm.is_finite() && chart_first_bpm > 0. {
        chart_first_bpm as f64
    } else {
        120.
    };
    let path = stable_path(&rows);
    let duration = env.time(env.values.len() - 1);
    let mut runs = Vec::new();
    let mut start = 0;
    while start < path.len() {
        let mut end = start + 1;
        while end < path.len() && path[end] == path[start] {
            end += 1;
        }
        let from = if start == 0 {
            0.
        } else {
            (rows[start - 1].time + rows[start].time) / 2.
        };
        let to = if end == path.len() {
            duration
        } else {
            (rows[end - 1].time + rows[end].time) / 2.
        };
        let margin = 4f64.min((to - from) / 5.);
        let offset = phase(&env, from + margin, to - margin, path[start] as f64);
        runs.push((from, to, path[start] as f64, offset));
        start = end;
    }
    let scale = chart_bpm / runs[0].2;
    let mut points = vec![TempoPoint {
        time: 0.,
        bpm: (runs[0].2 * scale) as f32,
    }];
    for i in 1..runs.len() {
        let previous = runs[i - 1];
        let current = runs[i];
        // Adjacent searches meet at a midpoint, so even a short intermediate
        // run cannot move its two boundaries past each other.
        let lower = if i == 1 {
            previous.0 + MIN_SEGMENT / 2.
        } else {
            (previous.0 + current.0) / 2.
        };
        let upper = if i + 1 == runs.len() {
            current.1 - MIN_SEGMENT / 2.
        } else {
            (current.0 + current.1) / 2.
        };
        let time =
            boundary(&env, current.0, previous.2, current.2, previous.3, current.3, lower, upper).max(points.last().unwrap().time as f64 + 0.0001);
        points.push(TempoPoint {
            time: time as f32,
            bpm: (current.2 * scale) as f32,
        });
    }
    Some(TempoEstimate {
        points,
        first_beat_time: runs[0].3 as f32,
        scale: scale as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    fn clicks(bpms: &[f64]) -> Vec<f32> {
        clicks_with_lengths(bpms, &vec![16.; bpms.len()])
    }

    fn clicks_with_lengths(bpms: &[f64], lengths: &[f64]) -> Vec<f32> {
        let duration = lengths.iter().sum::<f64>() + 0.125;
        let mut samples = vec![0.; (duration * RATE as f64) as usize];
        let mut start = 0.125;
        for (&bpm, &length) in bpms.iter().zip(lengths) {
            let end = start + length;
            let mut beat = start;
            while beat < end - 1e-6 {
                let index = (beat * RATE as f64).round() as usize;
                for j in 0..(RATE / 20).min(samples.len() - index) {
                    let time = j as f64 / RATE as f64;
                    samples[index + j] += ((-time / 0.012).exp() * (TAU * 1800. * time).sin()) as f32;
                }
                beat += 60. / bpm;
            }
            start = end;
        }
        samples
    }

    #[test]
    fn rhythm_auto_tempo_disabled_authored_and_silence_skip() {
        let reads = Cell::new(0);
        for (enabled, authored) in [(false, false), (true, true)] {
            assert!(infer_if_missing(
                enabled,
                authored,
                RATE * 8,
                RATE as u32,
                |_| {
                    reads.set(reads.get() + 1);
                    0.
                },
                120.
            )
            .is_none());
        }
        assert_eq!(reads.get(), 0);
        assert!(infer_if_missing(true, false, RATE * 8, RATE as u32, |_| 0., 120.).is_none());
    }

    #[test]
    fn rhythm_auto_tempo_recovers_120_180_90_and_continuous_time_nodes() {
        let audio = clicks(&[120., 180., 90.]);
        let map = infer_if_missing(true, false, audio.len(), RATE as u32, |i| audio[i], 120.).unwrap();
        let bpms: Vec<_> = map.points.iter().map(|point| point.bpm.round() as i32).collect();
        assert_eq!(bpms, vec![120, 180, 90], "{map:?}");
        assert!((map.points[1].time - 16.125).abs() < 0.5, "{map:?}");
        assert!((map.points[2].time - 32.125).abs() < 0.5, "{map:?}");
    }

    #[test]
    fn rhythm_auto_tempo_one_global_scale_keeps_seconds() {
        let audio = clicks(&[60., 100.]);
        let raw = infer_if_missing(true, false, audio.len(), RATE as u32, |i| audio[i], 60.).unwrap();
        let scaled = infer_if_missing(true, false, audio.len(), RATE as u32, |i| audio[i], 120.).unwrap();
        assert_eq!(scaled.points.iter().map(|point| point.bpm.round() as i32).collect::<Vec<_>>(), vec![120, 200], "{scaled:?}");
        assert!((scaled.scale - 2.).abs() < 1e-6, "{scaled:?}");
        assert_eq!(raw.points.len(), scaled.points.len());
        for (a, b) in raw.points.iter().zip(&scaled.points) {
            assert!((a.time - b.time).abs() < 1e-4);
            assert!((b.bpm - a.bpm * 2.).abs() < 1e-4);
        }
    }

    #[test]
    fn rhythm_auto_tempo_quiet_intro_accompaniment_and_fractional_lags_stay_stable() {
        for bpms in [[150., 200., 150.], [137., 173., 113.]] {
            let mut audio = clicks(&bpms);
            for (index, sample) in audio.iter_mut().enumerate() {
                let time = index as f64 / RATE as f64;
                if time < 16.125 {
                    *sample *= 0.12;
                }
                *sample += (0.045 * (TAU * 220. * time).sin() + 0.022 * (TAU * 329.63 * time).sin()) as f32;
            }
            let map = infer_if_missing(true, false, audio.len(), RATE as u32, |i| audio[i], bpms[0] as f32).unwrap();
            assert_eq!(map.points.iter().map(|point| point.bpm.round() as i32).collect::<Vec<_>>(), bpms.map(|bpm| bpm as i32), "{map:?}");
            assert!((map.scale - 1.).abs() < 1e-6, "{map:?}");
        }
    }

    #[test]
    fn rhythm_auto_tempo_sample_rate_conversion_and_unstructured_noise_are_safe() {
        let audio = clicks(&[60., 100.]);
        let source_rate = 44100;
        let count = audio.len() * source_rate / RATE;
        let map = infer_if_missing(
            true,
            false,
            count,
            source_rate as u32,
            |index| {
                let position = index as f64 * RATE as f64 / source_rate as f64;
                let left = position as usize;
                let mix = (position - left as f64) as f32;
                audio[left] * (1. - mix) + audio[(left + 1).min(audio.len() - 1)] * mix
            },
            120.,
        )
        .unwrap();
        assert_eq!(map.points.iter().map(|point| point.bpm.round() as i32).collect::<Vec<_>>(), vec![120, 200]);
        let mut noise = 42u32;
        assert!(infer_if_missing(
            true,
            false,
            RATE * 12,
            RATE as u32,
            |_| {
                noise = noise.wrapping_mul(1664525).wrapping_add(1013904223);
                (noise as f64 / u32::MAX as f64 - 0.5) as f32
            },
            120.
        )
        .is_none());
    }
    #[test]
    fn rhythm_auto_tempo_short_middle_run_nodes_remain_ordered() {
        let audio = clicks_with_lengths(&[120., 180., 90.], &[16., 8., 16.]);
        let map = infer_if_missing(true, false, audio.len(), RATE as u32, |i| audio[i], 120.).unwrap();
        assert_eq!(map.points.iter().map(|point| point.bpm.round() as i32).collect::<Vec<_>>(), vec![120, 180, 90], "{map:?}");
        for nodes in map.points.windows(2) {
            assert!(nodes[1].time > nodes[0].time, "{map:?}");
        }
        assert!((map.points[1].time - 16.125).abs() < 0.5, "{map:?}");
        assert!((map.points[2].time - 24.125).abs() < 0.5, "{map:?}");
    }
}
