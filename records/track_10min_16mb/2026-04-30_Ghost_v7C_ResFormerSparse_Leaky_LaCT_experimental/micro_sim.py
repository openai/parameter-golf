import sys, types, importlib.util, time, io
from pathlib import Path
spm = types.ModuleType('sentencepiece')
class FakeSP:
    def Load(self, path): return True
    def vocab_size(self): return 32
    def IdToPiece(self, i): return ('▁' if i % 7 == 0 else '') + chr(65 + (i % 26))
spm.SentencePieceProcessor = FakeSP
sys.modules['sentencepiece'] = spm
import torch
mod_path=Path('/mnt/data/ghost-v7c-resformer-leaky/records/track_10min_16mb/2026-04-30_11L_Ghost_v7C_LaCT_ResFormerSparse_Leaky_PATCHED/train_gpt.py')
spec=importlib.util.spec_from_file_location('train_gpt_patched', mod_path); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
class Args:
    vocab_size=32; num_layers=2; model_dim=32; num_heads=4; num_kv_heads=2; mlp_mult=2.0; bigram_buckets=64; bigram_dim=8; rope_dims=4; rope_base=10000.0; tie_embeddings=True; tied_embed_init_std=0.02; logit_softcap=30.0; qk_gain_init=2.0; xsa_layers=2; leaky_relu_slope=0.5; resformer_enabled=True; resformer_mode='all'; resformer_learned=True; resformer_detach_v0=True; use_mixed_quant=True; use_zstd=False; ttt_lr=1e-3; ttt_epochs=1; ttt_seq_len=8; ttt_batch_size=2; ttt_freeze_layers=0; ttt_no_qv=True; lact_chunk_size=2; eval_stride=2; eval_seq_len=8; train_seq_len=8; val_batch_size=16
args=Args(); device=torch.device('cpu'); torch.manual_seed(123); report=[]
def log(s): print(s, flush=True); report.append(str(s))
log('import_ok')
model=m.GPT(args.vocab_size,args.num_layers,args.model_dim,args.num_heads,args.num_kv_heads,args.mlp_mult,args.bigram_buckets,args.bigram_dim,args.rope_dims,args.rope_base,args.tie_embeddings,args.tied_embed_init_std,args.logit_softcap,args.qk_gain_init,args.xsa_layers,args.leaky_relu_slope,args.resformer_enabled,args.resformer_mode,args.resformer_learned,args.resformer_detach_v0).to(device)
m.restore_low_dim_params_to_fp32(model)
log(f'micro_model_params:{sum(p.numel() for p in model.parameters())}')
val_tokens=(torch.arange(2*args.ttt_seq_len+1,dtype=torch.int64)*5+1)%args.vocab_size
base_bytes_lut=torch.ones(args.vocab_size,dtype=torch.int32); has_leading_space_lut=torch.zeros(args.vocab_size,dtype=torch.bool); has_leading_space_lut[::7]=True; is_boundary_token_lut=torch.zeros(args.vocab_size,dtype=torch.bool)
x=val_tokens[:args.ttt_seq_len].unsqueeze(0); y=val_tokens[1:args.ttt_seq_len+1].unsqueeze(0)
t=time.time(); loss=model(x,y); log(f'forward_loss:{loss.detach().item():.6f} sec:{time.time()-t:.3f}')
per=model(x,y,return_per_token=True); log(f'per_token_shape:{tuple(per.shape)} mean_matches:{abs(float(per.mean().detach()-loss.detach()))<1e-6}')
# quick quant roundtrip on tiny model
t=time.time(); sd={k:v.detach().cpu() for k,v in model.state_dict().items()}; obj,raw_bytes=m.quantize_state_dict_mixed(sd,True); deq=m.dequantize_state_dict_mixed(obj); log(f'quant_done sec:{time.time()-t:.3f} raw_bytes:{raw_bytes}')
model2=m.GPT(args.vocab_size,args.num_layers,args.model_dim,args.num_heads,args.num_kv_heads,args.mlp_mult,args.bigram_buckets,args.bigram_dim,args.rope_dims,args.rope_base,args.tie_embeddings,args.tied_embed_init_std,args.logit_softcap,args.qk_gain_init,args.xsa_layers,args.leaky_relu_slope,args.resformer_enabled,args.resformer_mode,args.resformer_learned,args.resformer_detach_v0).to(device); missing,unexpected=model2.load_state_dict(deq, strict=False); m.restore_low_dim_params_to_fp32(model2); loss2=model2(x,y); log(f'roundtrip_loss:{loss2.detach().item():.6f} delta:{(loss2-loss).detach().item():.6f} missing:{len(missing)} unexpected:{len(unexpected)}')
t=time.time(); sl_loss,sl_bpb=m.eval_val_sliding(args,model2,0,1,device,val_tokens,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut,stride=args.eval_stride,seq_len=args.eval_seq_len,batch_seqs=1); log(f'sliding_eval_loss:{sl_loss:.6f} bpb:{sl_bpb:.6f} sec:{time.time()-t:.3f}')

# The container's torch.optim.AdamW import path can hang in this sandbox; RunPod should use real AdamW.
# For this CPU micro-sim only, patch in a minimal SGD-like optimizer to verify the legal
# score-first TTT control flow and parameter-freeze masks without depending on optimizer internals.
class _MicroAdamW:
    def __init__(self, params, lr=1e-3, **kwargs):
        self.params=list(params); self.lr=lr
    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none: p.grad=None
                else: p.grad.zero_()
    def step(self):
        with torch.no_grad():
            for p in self.params:
                if p.grad is not None:
                    p.add_(p.grad, alpha=-self.lr)
m.torch.optim.AdamW = _MicroAdamW

before={n:p.detach().clone() for n,p in model2.named_parameters()}; t=time.time(); ttt_loss,ttt_bpb=m.ttt_adapt(args,model2,val_tokens,device,log,base_bytes_lut,has_leading_space_lut,is_boundary_token_lut); log(f'ttt_loss:{ttt_loss:.6f} ttt_bpb:{ttt_bpb:.6f} sec:{time.time()-t:.3f}')
changed=[]; frozen_changed=[]
for n,p in model2.named_parameters():
    if not torch.equal(before[n], p.detach()):
        changed.append(n)
        if '.attn.c_q.' in n or '.attn.c_v.' in n:
            frozen_changed.append(n)
log(f'ttt_changed_params:{len(changed)} frozen_qv_changed:{len(frozen_changed)}')
buf=io.BytesIO(); torch.save(obj,buf); comp=m.compress(buf.getvalue(),False); raw=m.decompress(comp,False); log(f'zlib_compressed_bytes:{len(comp)} decompress_ok:{raw==buf.getvalue()}')
Path('/mnt/data/ghost_v7_micro_sim_report.txt').write_text('\n'.join(report), encoding='utf-8')
