"""Inline Path A v3 quantizer modifications into train_gpt_stacked_v2_fixed.py.

Path A v3 changes (validated yesterday + today):
1. Int8 per-tensor quant for control tensors (attn_scale, mlp_scale, resid_mix, skip_gates, skip_weights)
2. Int8 per-row quant for small matrices (bigram.proj, attn_gate_proj, smear_gate.weight)
3. dequantize_mixed updated to handle these new categories

Everything else unchanged.
"""
import re
F = 'train_gpt_stacked_v2_fixed.py'
src = open(F).read()

# Replace gptq_mixed_quantize
OLD_QUANT = """def gptq_mixed_quantize(state_dict,hessians,h):
\tresult={};meta={}
\tfor(name,tensor)in state_dict.items():
\t\tt=tensor.detach().cpu().contiguous()
\t\tif not t.is_floating_point()or t.numel()<=65536:result[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
\t\tif 'bigram.embed' in name:
\t\t\tbits=6;qmax=2**(bits-1)-1;row_max=t.abs().amax(dim=1,keepdim=True).clamp_min(1e-10);s=(row_max/qmax).squeeze(-1).to(torch.float16);sf=s.float().view(-1,1);q=torch.clamp(torch.round(t/sf),-qmax,qmax).to(torch.int8);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f'simple int{bits} (bigram embed)';continue
\t\tcs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f\"gptq (int{bits})\""""

NEW_QUANT = """def gptq_mixed_quantize(state_dict,hessians,h):
\tresult={};meta={}
\t_FORCE_INT8_SMALL=('bigram.proj','attn_gate_proj','smear_gate.weight')
\t_FORCE_INT8_PT=('attn_scale','mlp_scale','resid_mix','skip_gates','skip_weights')
\tfor(name,tensor)in state_dict.items():
\t\tt=tensor.detach().cpu().contiguous()
\t\tif not t.is_floating_point()or t.numel()<=65536:
\t\t\tif t.is_floating_point()and t.numel()>1 and any(k in name for k in _FORCE_INT8_PT):
\t\t\t\tma=t.abs().max().clamp_min(1e-10);sc=(ma/127.).float();q=torch.clamp(torch.round(t/sc),-127,127).to(torch.int8)
\t\t\t\tresult[name+'.q_pt']=q;result[name+'.scale_pt']=sc;meta[name]='pertensor int8 (control)';continue
\t\t\tif t.is_floating_point()and t.ndim==2 and any(k in name for k in _FORCE_INT8_SMALL):
\t\t\t\trm=t.abs().amax(dim=1,keepdim=True).clamp_min(1e-10);s=(rm/127.).squeeze(-1).to(torch.float16);sf=s.float().view(-1,1)
\t\t\t\tq=torch.clamp(torch.round(t/sf),-127,127).to(torch.int8);result[name+'.q']=q;result[name+'.scale']=s;meta[name]='simple int8 (small matrix)';continue
\t\t\tresult[name]=t.to(torch.float16)if t.is_floating_point()else t;meta[name]='passthrough (float16)';continue
\t\tif 'bigram.embed' in name:
\t\t\tbits=6;qmax=2**(bits-1)-1;row_max=t.abs().amax(dim=1,keepdim=True).clamp_min(1e-10);s=(row_max/qmax).squeeze(-1).to(torch.float16);sf=s.float().view(-1,1);q=torch.clamp(torch.round(t/sf),-qmax,qmax).to(torch.int8);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f'simple int{bits} (bigram embed)';continue
\t\tcs=h.embed_clip_sigmas if'tok_emb'in name else h.matrix_clip_sigmas;bits=h.embed_bits if'tok_emb'in name else h.matrix_bits;q,s=gptq_quantize_weight(t,hessians[name],clip_sigmas=cs,clip_range=2**(bits-1)-1);result[name+'.q']=q;result[name+'.scale']=s;meta[name]=f\"gptq (int{bits})\""""

if OLD_QUANT not in src:
    print("ERR: OLD_QUANT pattern not found")
    raise SystemExit(1)
src = src.replace(OLD_QUANT, NEW_QUANT, 1)
print("Replaced gptq_mixed_quantize with Path A v3 version")

# Replace dequantize_mixed
OLD_DEQ = """def dequantize_mixed(result,meta,template_sd):
\tout={}
\tfor(name,orig)in template_sd.items():
\t\tinfo=meta.get(name)
\t\tif info is None:continue
\t\torig_dtype=orig.dtype
\t\tif'passthrough'in info:
\t\t\tt=result[name]
\t\t\tif t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
\t\t\tout[name]=t;continue
\t\tq,s=result[name+'.q'],result[name+'.scale']
\t\tif s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
\t\telse:out[name]=(q.float()*float(s.item())).to(orig_dtype)
\treturn out"""

NEW_DEQ = """def dequantize_mixed(result,meta,template_sd):
\tout={}
\tfor(name,orig)in template_sd.items():
\t\tinfo=meta.get(name)
\t\tif info is None:continue
\t\torig_dtype=orig.dtype
\t\tif'passthrough'in info:
\t\t\tt=result[name]
\t\t\tif t.dtype==torch.float16 and orig_dtype in(torch.float32,torch.bfloat16):t=t.to(orig_dtype)
\t\t\tout[name]=t;continue
\t\tif'pertensor'in info:
\t\t\tq=result[name+'.q_pt'];sc=result[name+'.scale_pt']
\t\t\tout[name]=(q.float()*sc.float()).to(orig_dtype);continue
\t\tq,s=result[name+'.q'],result[name+'.scale']
\t\tif s.ndim>0:out[name]=(q.float()*s.float().view(q.shape[0],*[1]*(q.ndim-1))).to(orig_dtype)
\t\telse:out[name]=(q.float()*float(s.item())).to(orig_dtype)
\treturn out"""

if OLD_DEQ not in src:
    print("ERR: OLD_DEQ pattern not found")
    raise SystemExit(1)
src = src.replace(OLD_DEQ, NEW_DEQ, 1)
print("Replaced dequantize_mixed with Path A v3 version")

# Save
open(F, 'w').write(src)
import py_compile
py_compile.compile(F, doraise=True)
print(f"Saved + syntax OK: {len(src)} bytes")
