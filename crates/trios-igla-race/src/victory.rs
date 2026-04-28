use crate::IGLA_TARGET_BPB;

pub fn is_victory(bpb: f64) -> bool {
    bpb < IGLA_TARGET_BPB
}
