use crate::{AVCodecContext, AVFormatContext, AVFrame, AVPacket, AVPixelFormat, AVRational, StreamFormat, SwsContext};
use anyhow::{Context, Result};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
};
use tracing::error;

pub struct Video {
    stream_format: StreamFormat,
    frame_rate: AVRational,
    duration: Option<f64>,

    dropped: Arc<AtomicBool>,
    ended: AtomicBool,

    mutex: Arc<(Mutex<Option<Option<DecodedFrame>>>, Condvar)>,
    decode_thread: Option<JoinHandle<()>>,
}

struct DecodedFrame {
    image: &'static AVFrame,
    timestamp: Option<f64>,
}

impl Video {
    pub fn open(file: impl AsRef<str>, pix_fmt: AVPixelFormat) -> Result<Self> {
        let mut format_ctx = AVFormatContext::new()?;
        format_ctx.open_input(file.as_ref())?;
        format_ctx.find_stream_info()?;

        let video_stream = format_ctx.streams().into_iter().find(|it| it.is_video()).context("no video")?;
        let frame_rate = video_stream.frame_rate();
        let time_base = video_stream.time_base();
        let duration = video_stream.duration().or_else(|| format_ctx.duration());

        let decoder = video_stream.find_decoder()?;
        let mut codec_ctx = AVCodecContext::new(decoder, video_stream.codec_params(), Some(pix_fmt))?;

        let out_format = StreamFormat {
            pix_fmt,
            ..codec_ctx.stream_format()
        };

        let mutex = Arc::new((Mutex::new(None), Condvar::new()));

        let mut sws = SwsContext::new(codec_ctx.stream_format(), out_format.clone())?;
        let mut in_frame = AVFrame::new()?;
        let mut out_frame = AVFrame::new()?;
        out_frame.get_buffer(&out_format).context("failed to get buffer")?;

        let dropped = Arc::new(AtomicBool::default());

        let decode_thread = std::thread::spawn({
            let mut packet = AVPacket::new()?;
            let video_index = video_stream.index();
            let mutex = Arc::clone(&mutex);
            let dropped = Arc::clone(&dropped);
            move || {
                let mut decode_main = {
                    let mutex = Arc::clone(&mutex);
                    move || -> Result<()> {
                        while !dropped.load(Ordering::Relaxed) {
                            let eof = !format_ctx.read_frame(&mut packet)?;
                            if eof {
                                codec_ctx.send_eof()?;
                            } else {
                                let result = if packet.stream_index() == video_index {
                                    codec_ctx.send_packet(&packet)
                                } else {
                                    packet.unref();
                                    continue;
                                };
                                packet.unref();
                                result?;
                            }

                            while codec_ctx.receive_frame(&mut in_frame)? {
                                sws.scale(&in_frame, &mut out_frame);
                                let mut frame = mutex.0.lock().unwrap();
                                let timestamp = in_frame.best_effort_timestamp()
                                    .map(|pts| pts as f64 * time_base)
                                    .filter(|pts| pts.is_finite());
                                *frame = Some(Some(DecodedFrame {
                                    image: unsafe { std::mem::transmute(&out_frame) },
                                    timestamp,
                                }));
                                mutex.1.notify_one();
                                while frame.is_some() {
                                    if dropped.load(Ordering::Relaxed) {
                                        return Ok(());
                                    }
                                    frame = mutex.1.wait(frame).unwrap();
                                }
                            }
                            if eof {
                                break;
                            }
                        }
                        let mut frame = mutex.0.lock().unwrap();
                        *frame = Some(None);
                        mutex.1.notify_one();
                        Ok(())
                    }
                };
                if let Err(err) = decode_main() {
                    error!("decode failed: {err:?}");
                    let mut frame = mutex.0.lock().unwrap();
                    *frame = Some(None);
                    mutex.1.notify_one();
                }
            }
        });

        Ok(Self {
            stream_format: out_format,
            frame_rate,
            duration,

            dropped,
            ended: AtomicBool::default(),

            mutex,
            decode_thread: Some(decode_thread),
        })
    }

    pub fn stream_format(&self) -> StreamFormat {
        self.stream_format.clone()
    }

    pub fn frame_rate(&self) -> AVRational {
        AVRational { num: self.frame_rate.num, den: self.frame_rate.den }
    }

    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    pub fn with_frame<R>(&self, f: impl FnOnce(&AVFrame) -> R) -> Option<R> {
        self.with_timed_frame(|frame, _| f(frame))
    }

    pub fn with_timed_frame<R>(&self, f: impl FnOnce(&AVFrame, Option<f64>) -> R) -> Option<R> {
        let mut frame = self.mutex.0.lock().unwrap();
        loop {
            let Some(data) = frame.as_ref() else {
                frame = self.mutex.1.wait(frame).unwrap();
                continue;
            };
            let Some(data) = data else {
                self.ended.store(true, Ordering::SeqCst);
                return None;
            };
            let res = f(data.image, data.timestamp);
            *frame = None;
            self.mutex.1.notify_one();
            break Some(res);
        }
    }
}

impl Drop for Video {
    fn drop(&mut self) {
        self.dropped.store(true, Ordering::Relaxed);
        {
            let _guard = self.mutex.0.lock().unwrap();
            self.mutex.1.notify_one();
        }
        if let Some(handle) = self.decode_thread.take() {
            handle.join().unwrap();
        }
    }
}
