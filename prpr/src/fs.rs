use crate::{ext::spawn_task, info::ChartInfo};
use anyhow::{anyhow, bail, Context, Result};
use async_trait::async_trait;
use chardetng::EncodingDetector;
use concat_string::concat_string;
use macroquad::prelude::load_file;
use serde::Deserialize;
//use serde_json::Value;
use std::{
    any::Any,
    collections::HashMap,
    fs,
    io::{Cursor, Read, Seek, Write},
    path::Path,
    sync::{Arc, Mutex},
};
use tracing::warn;
use zip::{write::FileOptions, CompressionMethod, ZipArchive, ZipWriter};

pub fn update_zip<R: Read + Seek>(zip: &mut ZipArchive<R>, patches: HashMap<String, Vec<u8>>) -> Result<Vec<u8>> {
    let mut buffer = Vec::new();
    let mut w = ZipWriter::new(Cursor::new(&mut buffer));
    let options = FileOptions::default()
        .compression_method(CompressionMethod::Deflated)
        .unix_permissions(0o755);
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i).unwrap();
        let path = match entry.enclosed_name() {
            Some(path) => path.to_owned(),
            None => continue,
        };
        let path = path.display().to_string();
        if entry.is_dir() {
            w.add_directory(path, options)?;
        } else if !patches.contains_key(&path) {
            w.start_file(&path, options)?;
            std::io::copy(&mut entry, &mut w)?;
        }
    }
    for (path, data) in patches.into_iter() {
        w.start_file(path, options)?;
        w.write_all(&data)?;
    }
    w.finish()?;
    drop(w);
    Ok(buffer)
}

#[async_trait]
pub trait FileSystem: Send {
    async fn load_file(&mut self, path: &str) -> Result<Vec<u8>>;
    async fn exists(&mut self, path: &str) -> Result<bool>;
    fn list_root(&self) -> Result<Vec<String>>;
    fn clone_box(&self) -> Box<dyn FileSystem>;
    fn as_any(&mut self) -> &mut dyn Any;
}

#[derive(Clone)]
pub struct AssetsFileSystem(String);

#[async_trait]
impl FileSystem for AssetsFileSystem {
    async fn load_file(&mut self, path: &str) -> Result<Vec<u8>> {
        Ok(load_file(&concat_string!(self.0, path)).await?)
    }

    async fn exists(&mut self, path: &str) -> Result<bool> {
        // unlikely to be called
        Ok(load_file(&concat_string!(self.0, path)).await.is_ok())
    }

    fn list_root(&self) -> Result<Vec<String>> {
        Ok(Vec::new())
    }

    fn clone_box(&self) -> Box<dyn FileSystem> {
        Box::new(self.clone())
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct ExternalFileSystem(pub Arc<crate::dir::Dir>);

#[async_trait]
impl FileSystem for ExternalFileSystem {
    async fn load_file(&mut self, path: &str) -> Result<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            unimplemented!("cannot use external file system on wasm32")
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut file = self.0.open(path)?;
            tokio::task::spawn_blocking(move || {
                let mut res = Vec::new();
                file.read_to_end(&mut res)?;
                Ok(res)
            })
            .await?
        }
    }

    async fn exists(&mut self, path: &str) -> Result<bool> {
        self.0.exists(path)
    }

    fn list_root(&self) -> Result<Vec<String>> {
        Ok(self.0.read_dir(".")?.filter_map(|res| res.ok()?.file_name().into_string().ok()).collect())
    }

    fn clone_box(&self) -> Box<dyn FileSystem> {
        Box::new(self.clone())
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

#[derive(Clone)]
pub struct ZipFileSystem(pub Arc<Mutex<ZipArchive<Cursor<Vec<u8>>>>>, String);

fn is_zip_metadata(path: &str) -> bool {
    path.split('/').any(|part| part == "__MACOSX" || part == ".DS_Store" || part.starts_with("._"))
}

impl ZipFileSystem {
    pub fn new(bytes: Vec<u8>) -> Result<Self> {
        let zip = ZipArchive::new(Cursor::new(bytes))?;
        // ZIPs need not contain directory entries. Infer the common directory
        // from actual files, excluding Finder's resource forks and metadata.
        // A file in the archive root keeps the root empty even when there is
        // exactly one explicit subdirectory (e.g. chart.json + images/).
        let mut files = zip.file_names().filter(|name| !name.ends_with('/') && !is_zip_metadata(name));
        let mut root = files
            .next()
            .and_then(|name| name.rfind('/').map(|index| name[..=index].to_owned()))
            .unwrap_or_default();
        for name in files {
            while !name.starts_with(&root) {
                let parent_end = root[..root.len() - 1].rfind('/').map_or(0, |index| index + 1);
                root.truncate(parent_end);
            }
        }
        Ok(Self(Arc::new(Mutex::new(zip)), root))
    }
}

#[async_trait]
impl FileSystem for ZipFileSystem {
    async fn load_file(&mut self, path: &str) -> Result<Vec<u8>> {
        let arc = Arc::clone(&self.0);
        let path = concat_string!(self.1, path);
        spawn_task(move || {
            let mut zip = arc.lock().unwrap();
            let mut entry = zip.by_name(&path)?;
            let mut res = Vec::new();
            entry.read_to_end(&mut res)?;
            Ok(res)
        })
        .await
    }

    async fn exists(&mut self, path: &str) -> Result<bool> {
        Ok(self.0.lock().unwrap().by_name(&concat_string!(self.1, path)).is_ok())
    }

    fn list_root(&self) -> Result<Vec<String>> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .file_names()
            .filter_map(|name| name.strip_prefix(&self.1))
            .filter(|name| !name.is_empty() && !name.contains('/') && !is_zip_metadata(name))
            .map(str::to_owned)
            .collect())
    }

    fn clone_box(&self) -> Box<dyn FileSystem> {
        Box::new(self.clone())
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

pub struct PatchedFileSystem(pub Box<dyn FileSystem>, pub HashMap<String, Vec<u8>>);

#[async_trait]
impl FileSystem for PatchedFileSystem {
    async fn load_file(&mut self, path: &str) -> Result<Vec<u8>> {
        if let Some(data) = self.1.get(path) {
            Ok(data.clone())
        } else {
            self.0.load_file(path).await
        }
    }

    async fn exists(&mut self, path: &str) -> Result<bool> {
        Ok(self.0.exists(path).await? || self.1.contains_key(path))
    }

    fn list_root(&self) -> Result<Vec<String>> {
        let mut res = self.0.list_root()?;
        res.extend(self.1.keys().cloned());
        res.dedup();
        Ok(res)
    }

    fn clone_box(&self) -> Box<dyn FileSystem> {
        unimplemented!()
    }

    fn as_any(&mut self) -> &mut dyn Any {
        self
    }
}

fn infer_diff(info: &mut ChartInfo, level: &str) {
    if let Ok(val) = level
        .chars()
        .rev()
        .take_while(|it| it.is_ascii_digit())
        .collect::<String>()
        .chars()
        .rev()
        .collect::<String>()
        .parse::<u32>()
    {
        info.difficulty = val as f32;
    }
}

fn info_from_kv<'a>(it: impl Iterator<Item = (&'a str, String)>, csv: bool) -> Result<ChartInfo> {
    let mut info = ChartInfo::default();
    for (key, value) in it {
        if csv && value.trim().is_empty() {
            continue;
        }
        let key = key.trim();
        if key == "Path" {
            continue;
        }
        if key == "Level" {
            infer_diff(&mut info, &value);
        }
        if key == "AspectRatio" {
            // Old archives may carry this field; the viewport determines shape.
            continue;
        }
        if key == "BackgroundDim" {
            info.background_dim = value.parse().context("invalid background dim")?;
            continue;
        }
        if key == "NoteScale" || key == "ScaleRatio" {
            warn!("note scale is ignored");
            continue;
        }
        if key == "GlobalAlpha" {
            warn!("global alpha is ignored");
            continue;
        }
        //RPE160
        let mut deprecate = String::new();
        *match key {
            "Name" => &mut info.name,
            "Music" | "Song" => &mut info.music,
            "Chart" => &mut info.chart,
            "Image" | "Picture" => &mut info.illustration,
            "Level" => &mut info.level,
            "Illustrator" => &mut info.illustrator,
            "Artist" | "Composer" | "Musician" => &mut info.composer,
            "Charter" | "Designer" => &mut info.charter,
            //_ => bail!("Unknown key: {key}"),
            "LastEditTime" => &mut deprecate,
            "Length" => &mut deprecate,
            "EditTime" => &mut deprecate,
            _ => &mut deprecate,
        } = value;
    }
    Ok(info)
}

fn info_from_txt(text: &str) -> Result<ChartInfo> {
    // 跳过空行
    let mut it = text.lines().filter(|it| {!it.is_empty()}).peekable();
    let first = it.next();
    if first != Some("#") && first != Some("\u{feff}#") {
        bail!("expected the first line to be #");
    }
    let kvs = it
        .map(|line| -> Result<(&str, String)> {
            let Some((key, value)) = line.split_once(": ") else {
            bail!("expected \"Key: Value\"");
        };
            Ok((key, value.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    info_from_kv(kvs.into_iter(), false)
}

fn info_from_csv(text: &str) -> Result<ChartInfo> {
    let mut reader = csv::ReaderBuilder::new().flexible(true).from_reader(Cursor::new(text));
    // shitty design
    let headers = reader.headers()?.iter().map(str::to_owned).collect::<Vec<_>>();
    let record = reader.into_records().last().ok_or_else(|| anyhow!("expected csv records"))??; // ??
    info_from_kv(
        headers
            .iter()
            .zip(record.into_iter())
            .map(|(key, value)| (key.as_str(), value.to_owned())),
        true,
    )
}

const CHART_EXTS: &[&str] = &["json", "pec"];
const IMAGE_EXTS: &[&str] = &["png", "jpg", "jpeg", "bmp", "gif", "webp", "avif", "ppm"];
const AUDIO_EXTS: &[&str] = &["mp3", "ogg", "wav", "flac", "aac", "m4a"];
const INFO_EXTS: &[&str] = &["yml", "yaml"];

fn extension_of(file: &str) -> Option<String> {
    file.rsplit_once('.').map(|(_, ext)| ext.to_ascii_lowercase())
}

/// `extra.json` is chart metadata, not a chart itself.
fn is_chart_auxiliary(file: &str) -> bool {
    file.eq_ignore_ascii_case("extra.json")
}

/// Exactly one candidate is fine, none means "not found", more than one is an
/// error for files that must be selected unambiguously (charts, audio, info).
fn pick_unique(desc: &str, mut candidates: Vec<String>) -> Result<Option<String>> {
    candidates.sort();
    candidates.dedup();
    match candidates.len() {
        0 => Ok(None),
        1 => Ok(candidates.pop()),
        n => bail!(
            "found {n} candidate {desc} files ({}); expected exactly one - \
             rename or remove the extras, or name the file to use in info.yml",
            candidates.join(", ")
        ),
    }
}

async fn find_unique_ext(fs: &mut dyn FileSystem, desc: &str, exts: &[&str], keep: impl Fn(&str) -> bool) -> Result<Option<String>> {
    let files = fs.list_root().context("cannot list files")?;
    let candidates = files
        .into_iter()
        .filter(|file| keep(file))
        .filter(|file| extension_of(file).is_some_and(|ext| exts.contains(&ext.as_str())))
        .collect();
    pick_unique(desc, candidates)
}

async fn resolve_media(fs: &mut dyn FileSystem, info: &mut ChartInfo, meta_music: Option<String>, meta_illustration: Option<String>) -> Result<()> {
    if !fs.exists(&info.music).await? {
        if let Some(name) = meta_music.filter(|it| !it.is_empty()) {
            if fs.exists(&name).await? {
                info.music = name;
            }
        }
    }
    if !fs.exists(&info.music).await? {
        if let Some(name) = find_unique_ext(fs, "audio", AUDIO_EXTS, |_| true).await? {
            info.music = name;
        }
    }

    if !fs.exists(&info.illustration).await? {
        if let Some(name) = meta_illustration.filter(|it| !it.is_empty()) {
            if fs.exists(&name).await? {
                info.illustration = name;
            }
        }
    }
    if !fs.exists(&info.illustration).await? {
        // Charts may contain many image textures alongside their illustration.
        // With no declared illustration, choose a stable fallback by file name.
        let image = fs
            .list_root()
            .context("cannot list files")?
            .into_iter()
            .filter(|file| extension_of(file).is_some_and(|ext| IMAGE_EXTS.contains(&ext.as_str())))
            .min();
        if let Some(name) = image {
            info.illustration = name;
        }
    }
    Ok(())
}

/// Resolve file names for an info file read from disk: keep a declared name
/// when that file exists, otherwise match by extension. Chart and audio matches
/// must be unique; illustration images may coexist with chart textures.
pub async fn resolve_files(fs: &mut dyn FileSystem, info: &mut ChartInfo) -> Result<()> {
    if !fs.exists(&info.chart).await? {
        if let Some(chart) = find_unique_ext(fs, "chart", CHART_EXTS, |file| !is_chart_auxiliary(file)).await? {
            info.chart = chart;
        }
    }
    resolve_media(fs, info, None, None).await
}

pub async fn fix_info(fs: &mut dyn FileSystem, info: &mut ChartInfo) -> Result<()> {
    if !fs.exists(&info.chart).await? {
        let chart = find_unique_ext(fs, "chart", CHART_EXTS, |file| !is_chart_auxiliary(file))
            .await?
            .ok_or_else(|| anyhow!("cannot find chart (*.json or *.pec)"))?;
        info.chart = chart;
    }

    let mut meta_music = None;
    let mut meta_illustration = None;
    if let Ok(bytes) = fs.load_file(&info.chart).await {
        if let Ok((mut value, _s)) = tokio::task::spawn_blocking(move || {
            let s = String::from_utf8(bytes)?;
            let json: serde_json::Value = serde_json::from_str(&s)?;
            Ok::<_, anyhow::Error>((json, s))
        })
        .await?
        {
            #[derive(Deserialize)]
            struct RPEMeta {
                name: String,
                level: String,
                background: String,
                charter: String,
                composer: Option<String>,
                illustrator: Option<String>,
                song: String,
            }

            if let Ok(meta) = serde_json::from_value::<RPEMeta>(value["META"].take()) {
                info.name = meta.name;
                infer_diff(info, &meta.level);
                info.level = meta.level;
                info.charter = meta.charter;
                if let Some(val) = meta.composer {
                    info.composer = val;
                }
                if let Some(val) = meta.illustrator {
                    info.illustrator = val;
                }
                meta_illustration = Some(meta.background);
                meta_music = Some(meta.song);
            }
        }
    }
    resolve_media(fs, info, meta_music, meta_illustration).await?;
    Ok(())
}

fn bytes_to_text_auto(data: &[u8]) -> String {
    let mut det = EncodingDetector::new();
    det.feed(data, true);
    let encoding = det.guess(None, true);
    let (s, _, _) = encoding.decode(data);
    s.into_owned()
}

pub async fn load_info(fs: &mut dyn FileSystem) -> Result<ChartInfo> {
    let mut info_opt = None;
    for filename in &[":info", "info.yml", "info.txt", "info.csv"] {
        if let Ok(bytes) = fs.load_file(filename).await {
            let text = bytes_to_text_auto(&bytes);
            info_opt = Some(match *filename {
                ":info" | "info.yml" => serde_yaml::from_str(&text)?,
                "info.txt" => info_from_txt(&text)?,
                "info.csv" => info_from_csv(&text)?,
                _ => unreachable!(),
            });
            break;
        }
    }

    // Nothing explicitly named: accept the only *.yml / *.yaml in the root,
    // whatever it is called. More than one is ambiguous, not "pick the first".
    if info_opt.is_none() {
        if let Some(name) = find_unique_ext(fs, "description", INFO_EXTS, |_| true).await? {
            let bytes = fs.load_file(&name).await?;
            let text = bytes_to_text_auto(&bytes);
            info_opt = Some(serde_yaml::from_str(&text)?);
        }
    }

    let inferred = info_opt.is_none();
    let mut info = info_opt.unwrap_or_default();
    if inferred {
        warn!("none of info.yml, info.txt and info.csv is found, inferring");
        fix_info(fs, &mut info).await?;
    } else {
        resolve_files(fs, &mut info).await?;
    }
    Ok(info)
}

pub fn fs_from_file(path: &Path) -> Result<Box<dyn FileSystem + Send + Sync + 'static>> {
    let meta = fs::metadata(path)?;
    Ok(if meta.is_file() {
        let bytes = fs::read(path).with_context(|| format!("failed to read from {}", path.display()))?;
        Box::new(ZipFileSystem::new(bytes).with_context(|| format!("cannot open {} as zip archive", path.display()))?)
    } else {
        Box::new(ExternalFileSystem(Arc::new(crate::dir::Dir::new(path)?)))
    })
}

pub fn fs_from_assets(name: impl Into<String>) -> Result<Box<dyn FileSystem + Send + Sync + 'static>> {
    Ok(Box::new(AssetsFileSystem(name.into())))
}

#[cfg(test)]
mod zip_tests {
    use super::*;
    use std::{path::PathBuf, sync::atomic::{AtomicUsize, Ordering}};

    struct TestDirectory(PathBuf);

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn media_filesystems(entries: &[(&str, &[u8])]) -> (TestDirectory, Vec<Box<dyn FileSystem>>) {
        static NEXT_DIRECTORY: AtomicUsize = AtomicUsize::new(0);
        let directory = TestDirectory(std::env::temp_dir().join(format!(
            "prpr-media-{}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos(),
            NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed),
        )));
        std::fs::create_dir(&directory.0).unwrap();
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        let options = FileOptions::default().compression_method(CompressionMethod::Deflated);
        for &(name, bytes) in entries {
            let path = directory.0.join(name);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, bytes).unwrap();
            writer.start_file(name, options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        let archive = writer.finish().unwrap().into_inner();
        let filesystems: Vec<Box<dyn FileSystem>> = vec![
            Box::new(ZipFileSystem::new(archive).unwrap()),
            fs_from_file(&directory.0).unwrap(),
        ];
        (directory, filesystems)
    }

    fn check_media(entries: &[(&str, &[u8])], illustration: &str) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let (_directory, filesystems) = media_filesystems(entries);
        for mut fs in filesystems {
            runtime.block_on(async {
                let info = load_info(fs.as_mut()).await.unwrap();
                assert_eq!(info.illustration, illustration);
                for &(name, bytes) in entries {
                    assert!(fs.exists(name).await.unwrap());
                    assert_eq!(fs.load_file(name).await.unwrap(), bytes, "file: {name}");
                    assert_eq!(fs.clone_box().load_file(name).await.unwrap(), bytes, "cloned file: {name}");
                }
            });
        }
    }

    #[test]
    fn multiple_images_are_preserved_and_fallback_is_stable() {
        let mut entries: Vec<(&str, &[u8])> = vec![
            ("z-texture.png", b"root texture"),
            ("images/deep/texture.webp", b"nested texture"),
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("a-cover.JPG", b"cover"),
            ("texture.bmp", b"another root texture"),
        ];
        check_media(&entries, "a-cover.JPG");
        entries.reverse();
        check_media(&entries, "a-cover.JPG");
    }

    #[test]
    fn declared_illustration_takes_priority_over_other_images() {
        check_media(&[
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("a-texture.png", b"texture"),
            ("z-cover.png", b"cover"),
            ("images/texture.webp", b"nested texture"),
            ("info.yml", b"chart: custom.pec\nmusic: music.ogg\nillustration: z-cover.png\n"),
        ], "z-cover.png");
    }

    #[test]
    fn missing_declared_illustration_accepts_multiple_images() {
        check_media(&[
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("z-texture.png", b"texture"),
            ("a-cover.png", b"cover"),
            ("info.yml", b"chart: custom.pec\nmusic: music.ogg\nillustration: missing.png\n"),
        ], "a-cover.png");
    }

    #[test]
    fn rpe_meta_illustration_takes_priority_over_other_images() {
        check_media(&[
            ("custom.json", br#"{"META":{"name":"Test","level":"IN 12","background":"z-cover.png","charter":"Test","song":"music.ogg"}}"#),
            ("music.ogg", b"music"),
            ("a-texture.png", b"texture"),
            ("z-cover.png", b"cover"),
            ("images/texture.webp", b"nested texture"),
        ], "z-cover.png");
    }

    #[test]
    fn default_illustration_takes_priority_over_other_images() {
        check_media(&[
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("a-texture.png", b"texture"),
            ("background.png", b"default cover"),
        ], "background.png");
    }

    #[test]
    fn single_image_is_still_inferred() {
        check_media(&[
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("cover.WeBp", b"cover"),
        ], "cover.WeBp");
    }

    #[test]
    fn missing_images_keep_the_default_illustration() {
        check_media(&[
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
        ], "background.png");
    }

    #[test]
    fn non_image_file_inference_still_rejects_ambiguity() {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let cases: [(&str, &[(&str, &[u8])]); 3] = [
            ("chart", &[("a.json", b"chart"), ("b.pec", b"chart")]),
            ("audio", &[("custom.pec", b"chart"), ("a.ogg", b"audio"), ("b.mp3", b"audio")]),
            ("description", &[("a.yml", b"name: A"), ("b.yaml", b"name: B")]),
        ];
        for (kind, entries) in cases {
            let (_directory, filesystems) = media_filesystems(entries);
            for mut fs in filesystems {
                let error = runtime.block_on(load_info(fs.as_mut())).err().unwrap();
                assert!(error.to_string().contains(&format!("candidate {kind} files")), "{error}");
                assert!(error.to_string().contains("expected exactly one"), "{error}");
            }
        }
    }

    fn archive(prefix: &str, directories: bool, finder_metadata: bool, declared_info: bool) -> Vec<u8> {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        let options = FileOptions::default().compression_method(CompressionMethod::Deflated);
        if directories {
            if !prefix.is_empty() {
                writer.add_directory(prefix, options).unwrap();
            }
            writer.add_directory(format!("{prefix}images/"), options).unwrap();
        }
        // First entry is deliberately deeper than the chart root.
        for (name, bytes) in [
            ("images/texture.png", b"texture".as_slice()),
            ("custom.pec", b"chart"),
            ("music.ogg", b"music"),
            ("cover.png", b"cover"),
        ] {
            writer.start_file(format!("{prefix}{name}"), options).unwrap();
            writer.write_all(bytes).unwrap();
        }
        if declared_info {
            writer.start_file(format!("{prefix}info.yml"), options).unwrap();
            writer.write_all(b"name: Test\nchart: custom.pec\nmusic: music.ogg\nillustration: cover.png\n").unwrap();
        }
        if finder_metadata {
            writer.add_directory("__MACOSX/", options).unwrap();
            for name in ["__MACOSX/._custom.pec", ".DS_Store", &format!("{prefix}._custom.pec"), &format!("{prefix}.DS_Store")] {
                writer.start_file(name, options).unwrap();
                writer.write_all(b"Finder metadata").unwrap();
            }
        }
        writer.finish().unwrap().into_inner()
    }

    fn check_layout(prefix: &str, directories: bool, finder_metadata: bool) {
        let runtime = tokio::runtime::Runtime::new().unwrap();
        for declared_info in [false, true] {
            let mut fs = ZipFileSystem::new(archive(prefix, directories, finder_metadata, declared_info)).unwrap();
            assert_eq!(fs.1, prefix);
            let mut listed = fs.list_root().unwrap();
            listed.sort();
            let mut expected = vec!["cover.png", "custom.pec", "music.ogg"];
            if declared_info {
                expected.push("info.yml");
                expected.sort();
            }
            assert_eq!(listed, expected);
            runtime.block_on(async {
                let info = load_info(&mut fs).await.unwrap();
                assert_eq!(info.chart, "custom.pec");
                assert_eq!(info.music, "music.ogg");
                assert_eq!(info.illustration, "cover.png");
                for (path, bytes) in [(info.chart, b"chart".as_slice()), (info.music, b"music"), (info.illustration, b"cover")] {
                    assert!(fs.exists(&path).await.unwrap());
                    assert_eq!(fs.load_file(&path).await.unwrap(), bytes);
                }
                assert_eq!(fs.load_file("images/texture.png").await.unwrap(), b"texture");
                assert_eq!(fs.clone_box().load_file("custom.pec").await.unwrap(), b"chart");
            });
        }
    }

    #[test]
    fn flat_archive_with_a_subdirectory_keeps_chart_root() {
        check_layout("", true, false);
    }

    #[test]
    fn wrapped_archive_lists_and_loads_relative_paths() {
        check_layout("wrapped/", true, false);
    }

    #[test]
    fn wrapped_archive_without_directory_entries_is_inferred() {
        check_layout("wrapped/", false, false);
    }

    #[test]
    fn finder_archive_ignores_metadata_and_resource_forks() {
        check_layout("谱面包/", true, true);
        check_layout("", false, true);
    }

    #[test]
    fn multiple_wrapper_directories_are_inferred() {
        check_layout("outer/wrapped/", false, true);
    }

    #[test]
    fn metadata_only_archive_has_no_chart_root() {
        let mut writer = ZipWriter::new(Cursor::new(Vec::new()));
        let options = FileOptions::default();
        writer.add_directory("__MACOSX/", options).unwrap();
        writer.start_file("__MACOSX/._chart.json", options).unwrap();
        writer.write_all(b"resource fork").unwrap();
        writer.start_file(".DS_Store", options).unwrap();
        writer.write_all(b"metadata").unwrap();
        let mut fs = ZipFileSystem::new(writer.finish().unwrap().into_inner()).unwrap();
        assert!(fs.1.is_empty());
        assert!(fs.list_root().unwrap().is_empty());
        let error = tokio::runtime::Runtime::new().unwrap().block_on(load_info(&mut fs)).err().unwrap();
        assert!(error.to_string().contains("cannot find chart"));
    }
}
