extern fn fakeoutput_i32() -> *mut i32 {
    let fake = vec![10];
    fake.leak().as_mut_ptr()
}

pub extern fn fakeoutput_f32() -> *mut f32 {
    let fake = vec![10.0];
    fake.leak().as_mut_ptr()
}
