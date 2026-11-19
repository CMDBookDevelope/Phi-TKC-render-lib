use crate::{ffi, handle, AVResult, OwnedPtr, StreamFormat};
use anyhow::{Context, Result};

#[repr(transparent)]
pub struct AVFrame(pub(crate) OwnedPtr<ffi::AVFrame>);
impl AVFrame {
    pub fn new() -> Result<Self> {
        unsafe { Ok(Self(OwnedPtr::new(ffi::av_frame_alloc()).context("failed to allocate frame")?)) }
    }

    pub fn get_buffer(&mut self, format: &StreamFormat) -> AVResult<()> {
        unsafe {
            let this = self.0.as_mut();
            this.width = format.width;
            this.height = format.height;
            this.format = format.pix_fmt.0;
            handle(ffi::av_frame_get_buffer(self.0 .0, 1))
        }
    }

    // TODO: is this correct?
    pub fn data(&self, index: usize) -> &[u8] {
        unsafe {
            let this = self.0.as_ref();
            std::slice::from_raw_parts(this.data[index], this.linesize[index] as usize * this.height as usize)
        }
    }

    pub fn data_half(&self, index: usize) -> &[u8] {
        let data = self.data(index);
        &data[..data.len() / 2]
    }

    pub fn line_size(&self) -> i32 {
        unsafe { self.0.as_ref().linesize[0] }
    }

    pub(crate) fn best_effort_timestamp(&self) -> Option<i64> {
        let pts = unsafe { self.0.as_ref().best_effort_timestamp };
        (pts != i64::MIN).then_some(pts)
    }

    pub fn copy_plane(&self, index: usize, width: usize, height: usize, output: &mut [u8]) {
        let frame = unsafe { self.0.as_ref() };
        let stride = frame.linesize[index];
        assert!(stride >= width as i32 && !frame.data[index].is_null());
        assert_eq!(output.len(), width * height);
        for (row, dst) in output.chunks_exact_mut(width).enumerate() {
            let src = unsafe {
                std::slice::from_raw_parts(frame.data[index].add(row * stride as usize), width)
            };
            dst.copy_from_slice(src);
        }
    }
}

impl Drop for AVFrame {
    fn drop(&mut self) {
        unsafe {
            ffi::av_frame_free(self.0.as_self_mut());
        }
    }
}

unsafe impl Send for AVFrame {}
unsafe impl Sync for AVFrame {}
