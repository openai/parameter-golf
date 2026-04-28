pub fn check_phi_anchor() -> bool {
    let phi = (1.0_f64 + 5.0_f64.sqrt()) / 2.0;
    let anchor = phi * phi + 1.0 / (phi * phi);
    (anchor - 3.0).abs() < 1e-10
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phi_anchor() {
        assert!(check_phi_anchor());
    }
}
