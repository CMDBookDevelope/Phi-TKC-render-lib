use crate::{ffi, AVCodecParamsRef, AVCodecRef, AVRational};
use anyhow::Result;

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct AVStreamRef(*const ffi::AVStream);
impl AVStreamRef {
    pub fn index(&self) -> i32 {
        unsafe { (*self.0).index as i32 }
    }

    pub fn frame_rate(&self) -> AVRational {
        let stream = unsafe { &*self.0 };
        if stream.avg_frame_rate.num > 0 && stream.avg_frame_rate.den > 0 {
            stream.avg_frame_rate.into()
        } else {
            stream.r_frame_rate.into()
        }
    }

    pub fn time_base(&self) -> f64 {
        AVRational::from(unsafe { (*self.0).time_base }).to_f64()
    }

    pub fn duration(&self) -> Option<f64> {
        let duration = unsafe { (*self.0).duration } as f64 * self.time_base();
        (duration.is_finite() && duration > 0.).then_some(duration)
    }

    pub fn is_video(&self) -> bool {
        unsafe { (*(*self.0).codecpar).codec_type == 0 }
    }

    pub fn codec_params(&self) -> AVCodecParamsRef {
        AVCodecParamsRef(unsafe { (*self.0).codecpar })
    }

    pub fn find_decoder(&self) -> Result<AVCodecRef> {
        AVCodecRef::find_decoder(self.codec_params().codec_id())
    }
}
