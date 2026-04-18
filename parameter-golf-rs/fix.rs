use std::sync::Arc;
pub fn test() {
    let mut arc = Arc::new(5);
    let r = Arc::get_mut(&mut arc).unwrap();
    *r = 10;
}
