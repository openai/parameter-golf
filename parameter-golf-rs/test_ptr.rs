use cudarc::driver::{DeviceRepr, LaunchArgs, PushKernelArg};
#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct CudaPtr(pub u64);
unsafe impl DeviceRepr for CudaPtr {}
pub fn test_arg<'a>(mut args: LaunchArgs<'a>) {
    let p = CudaPtr(0x1234);
    args = args.arg(&p);
}
