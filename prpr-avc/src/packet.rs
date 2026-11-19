use crate::{ffi, OwnedPtr};
use anyhow::{Context, Result};

#[repr(transparent)]
pub struct AVPacket(pub(crate) OwnedPtr<ffi::AVPacket>);
impl AVPacket {
    pub fn new() -> Result<Self> {
        unsafe { Ok(Self(OwnedPtr::new(ffi::av_packet_alloc()).context("failed to allocate packet")?)) }
    }

    pub fn stream_index(&self) -> i32 {
        unsafe { self.0.as_ref().stream_index }
    }

    pub fn unref(&mut self) {
        unsafe { ffi::av_packet_unref(self.0 .0) }
    }
}

unsafe impl Send for AVPacket {}

impl Drop for AVPacket {
    fn drop(&mut self) {
        unsafe { ffi::av_packet_free(self.0.as_self_mut()) }
    }
}
