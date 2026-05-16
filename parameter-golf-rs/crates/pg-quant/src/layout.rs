use pg_core::error::{PgError, PgResult};
use pg_model::QuantSpec;
use pg_quant_macros::quant_layout;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompiledQuantLayout {
    pub name: &'static str,
    pub matrix_bits: u8,
    pub mlp_bits: u8,
    pub embed_bits: u8,
    pub attn_gate_bits: u8,
    pub gptq_calibration_batches: usize,
    pub target_artifact_bytes: usize,
    pub lqer_enabled: bool,
}

impl CompiledQuantLayout {
    pub fn validate(self) -> PgResult<Self> {
        for (name, bits) in [
            ("matrix_bits", self.matrix_bits),
            ("mlp_bits", self.mlp_bits),
            ("embed_bits", self.embed_bits),
            ("attn_gate_bits", self.attn_gate_bits),
        ] {
            if !(2..=8).contains(&bits) {
                return Err(PgError::InvalidOp(format!(
                    "compiled quant layout {} has unsupported {name}={bits}; expected 2..=8",
                    self.name
                )));
            }
        }
        Ok(self)
    }

    pub const fn uses_mixed_mlp_bits(self) -> bool {
        self.matrix_bits != self.mlp_bits
    }

    pub fn matches_quant_spec(self, quant_spec: &QuantSpec) -> bool {
        quant_spec.matrix_bits == self.matrix_bits
            && quant_spec.mlp_bits == self.mlp_bits
            && quant_spec.embed_bits == self.embed_bits
            && quant_spec.attn_gate_bits == self.attn_gate_bits
            && quant_spec.gptq_calibration_batches == self.gptq_calibration_batches
            && quant_spec.target_artifact_bytes == self.target_artifact_bytes
            && quant_spec.lqer.enabled == self.lqer_enabled
    }

    pub fn validate_quant_spec(self, quant_spec: &QuantSpec) -> PgResult<Self> {
        let layout = self.validate()?;
        if !layout.matches_quant_spec(quant_spec) {
            return Err(PgError::InvalidOp(format!(
                "compiled quant layout {} does not match QuantSpec: layout bits={}/{}/{} gate={} gptq_batches={} target_bytes={} lqer={}, spec bits={}/{}/{} gate={} gptq_batches={} target_bytes={} lqer={}",
                layout.name,
                layout.matrix_bits,
                layout.mlp_bits,
                layout.embed_bits,
                layout.attn_gate_bits,
                layout.gptq_calibration_batches,
                layout.target_artifact_bytes,
                layout.lqer_enabled,
                quant_spec.matrix_bits,
                quant_spec.mlp_bits,
                quant_spec.embed_bits,
                quant_spec.attn_gate_bits,
                quant_spec.gptq_calibration_batches,
                quant_spec.target_artifact_bytes,
                quant_spec.lqer.enabled
            )));
        }
        Ok(layout)
    }
}

pub const FRONTIER_1855_MIXED_INT5_INT6_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_1855_mixed_int5_int6",
    matrix_bits: 6,
    mlp_bits: 5,
    embed_bits: 7,
    attn_gate_bits: 8,
    gptq_calibration_batches: 16,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub const FRONTIER_2135_MIXED_INT5_INT4_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_2135_mixed_int5_int4",
    matrix_bits: 5,
    mlp_bits: 4,
    embed_bits: 6,
    attn_gate_bits: 8,
    gptq_calibration_batches: 32,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub const FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT: CompiledQuantLayout = quant_layout! {
    name: "frontier_2135_calib32_int6_int7",
    matrix_bits: 6,
    mlp_bits: 6,
    embed_bits: 7,
    attn_gate_bits: 8,
    gptq_calibration_batches: 32,
    target_artifact_bytes: 16_000_000,
    lqer_enabled: true,
};

pub fn compiled_layout_for_quant_spec(quant_spec: &QuantSpec) -> Option<CompiledQuantLayout> {
    [
        FRONTIER_1855_MIXED_INT5_INT6_LAYOUT,
        FRONTIER_2135_MIXED_INT5_INT4_LAYOUT,
        FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT,
    ]
    .into_iter()
    .find(|layout| layout.matches_quant_spec(quant_spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontier_layouts_are_compiled_and_valid() {
        let legacy = FRONTIER_1855_MIXED_INT5_INT6_LAYOUT.validate().unwrap();
        assert_eq!(legacy.matrix_bits, 6);
        assert_eq!(legacy.mlp_bits, 5);
        assert_eq!(legacy.embed_bits, 7);
        assert_eq!(legacy.gptq_calibration_batches, 16);
        assert!(legacy.uses_mixed_mlp_bits());
        assert!(legacy.lqer_enabled);

        let current = FRONTIER_2135_MIXED_INT5_INT4_LAYOUT.validate().unwrap();
        assert_eq!(current.matrix_bits, 5);
        assert_eq!(current.mlp_bits, 4);
        assert_eq!(current.embed_bits, 6);
        assert_eq!(current.gptq_calibration_batches, 32);
        assert!(current.uses_mixed_mlp_bits());
        assert!(current.lqer_enabled);

        let pr2135 = FRONTIER_2135_CALIB32_INT6_INT7_LAYOUT.validate().unwrap();
        assert_eq!(pr2135.matrix_bits, 6);
        assert_eq!(pr2135.mlp_bits, 6);
        assert_eq!(pr2135.embed_bits, 7);
        assert_eq!(pr2135.gptq_calibration_batches, 32);
        assert!(!pr2135.uses_mixed_mlp_bits());
        assert!(pr2135.lqer_enabled);
    }

    #[test]
    fn compiled_layout_matches_current_record_quant_spec() {
        let quant_spec = QuantSpec {
            matrix_bits: 5,
            mlp_bits: 4,
            embed_bits: 6,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let layout = compiled_layout_for_quant_spec(&quant_spec)
            .expect("2135 quant spec should have a compiled layout");
        assert_eq!(layout.name, "frontier_2135_mixed_int5_int4");
        layout.validate_quant_spec(&quant_spec).unwrap();
    }

    #[test]
    fn compiled_layout_matches_upstream_pr2135_quant_spec() {
        let quant_spec = QuantSpec {
            matrix_bits: 6,
            mlp_bits: 6,
            embed_bits: 7,
            attn_gate_bits: 8,
            gptq_calibration_batches: 32,
            lqer: pg_model::spec::LqerSpec {
                enabled: true,
                ..Default::default()
            },
            ..QuantSpec::default()
        };
        let layout = compiled_layout_for_quant_spec(&quant_spec)
            .expect("upstream PR2135 quant spec should have a compiled layout");
        assert_eq!(layout.name, "frontier_2135_calib32_int6_int7");
        layout.validate_quant_spec(&quant_spec).unwrap();
    }
}
