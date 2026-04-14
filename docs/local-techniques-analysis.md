# Parameter Golf Local Submissions Analysis
**Total submissions analyzed**: 24 (21 record + 3 non-record)
**Generated**: 2026-04-14 (automated technique extraction)

---

## Techniques by Frequency

- **Adam** — 23 submissions (96%)
- **Cosine Warmdown** — 23 submissions (96%)
- **EMA** — 23 submissions (96%)
- **GQA** — 23 submissions (96%)
- **Gradient Clipping** — 23 submissions (96%)
- **Int4** — 23 submissions (96%)
- **Int6** — 23 submissions (96%)
- **Int8** — 23 submissions (96%)
- **Logit Softcap** — 23 submissions (96%)
- **Multi-Head** — 23 submissions (96%)
- **Muon** — 23 submissions (96%)
- **RoPE** — 23 submissions (96%)
- **Tied Embeddings** — 23 submissions (96%)
- **Vocab Optimization** — 23 submissions (96%)
- **Warmup** — 23 submissions (96%)
- **Sliding Window** — 15 submissions (62%)
- **Zstd** — 12 submissions (50%)
- **QAT** — 11 submissions (46%)
- **Weight Decay** — 11 submissions (46%)
- **AdamW** — 10 submissions (42%)
- **SWA** — 9 submissions (38%)
- **BigramHash** — 9 submissions (38%)
- **SmearGate** — 8 submissions (33%)
- **XSA** — 7 submissions (29%)
- **NTK** — 6 submissions (25%)
- **Flash Attention** — 6 submissions (25%)
- **FP16 Embed** — 5 submissions (21%)
- **PartialRoPE** — 4 submissions (17%)
- **LoRA** — 3 submissions (12%)
- **TTT** — 2 submissions (8%)
- **GPTQ** — 2 submissions (8%)
- **LZMA** — 2 submissions (8%)
- **LeakyReLU** — 2 submissions (8%)
- **OrthoInit** — 2 submissions (8%)
- **SGD** — 1 submissions (4%)
- **TrigramHash** — 1 submissions (4%)
- **U-Net** — 1 submissions (4%)

---

## Techniques → Submissions Mapping

### Adam

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Cosine Warmdown

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### EMA

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### GQA

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Gradient Clipping

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Int4

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Int6

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Int8

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Logit Softcap

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Multi-Head

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Muon

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### RoPE

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Tied Embeddings

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Vocab Optimization

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Warmup

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_TrainingOptSeq4096` (record, BPB: 2.9266)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-19_10L_MixedPrecision` (record, BPB: 2.9590)
- `03-18_Quasi10Bfrom50B_SP1024_9x512_KV4_4h_pgut3` (non-record, BPB: 2.9411)
- `03-18_LowerLR` (record, BPB: 2.9791)
- `03-18_LongContextSeq2048` (record, BPB: 2.9372)
- `03-18_FP16Embed_WD3600` (record, BPB: 2.9712)
- `03-17_NaiveBaseline` (record, BPB: 2.9903)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Sliding Window

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### Zstd

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### QAT

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-19_Seq2048_FP16Emb_TunedLR` (record, BPB: 2.8223)
- `03-19_MixedQuant_Int6Int8_SlidingWindow` (record, BPB: 2.8330)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### Weight Decay

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### AdamW

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### SWA

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### BigramHash

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)

### SmearGate

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` (record, BPB: 2.7911)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-20_10L_Int5MLP_MuonWD04_SWA50` (record, BPB: 1.6487)

### XSA

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)

### NTK

- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` (record, BPB: None)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### Flash Attention

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)

### FP16 Embed

- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)
- `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` (record, BPB: 1.1271)
- `03-20_11L_EfficientPartialXSA_FA3_SWA120` (record, BPB: 2.7543)
- `03-19_WarmdownQuantization` (record, BPB: 2.8194)
- `03-19_MLP3x_QAT_Int6_SlidingWindow` (record, BPB: None)

### PartialRoPE

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` (record, BPB: 1.1233)
- `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` (record, BPB: 1.1248)

### LoRA

- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)
- `03-19_SlidingWindowEval` (record, BPB: 2.9048)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### TTT

- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)
- `03-17_LoRA_TTT` (record, BPB: 2.9059)

### GPTQ

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)

### LZMA

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)

### LeakyReLU

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)

### OrthoInit

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)
- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)

### SGD

- `03-23_LeakyReLU_LegalTTT_ParallelMuon` (record, BPB: None)

### TrigramHash

- `03-25_ValCalib_GPTQ_XSA_BigramHash3072` (record, BPB: 2.7154)

### U-Net

- `03-21_DepthRecurrence_MixedPrecisionQuant` (non-record, BPB: 5.8161)


---

## Record Submissions (10min_16mb) — Chronological

| Date | Submission | BPB | Techniques |
|------|-----------|-----|------------|
| 2026-03-25 | `03-25_ValCalib_GPTQ_XSA_BigramHash3072` | 2.7154 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +26 |
| 2026-03-23 | `03-23_LeakyReLU_LegalTTT_ParallelMuon` | None | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +27 |
| 2026-03-22 | `03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233` | 1.1233 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +21 |
| 2026-03-21 | `03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248` | 1.1248 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +23 |
| 2026-03-20 | `03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA` | 2.7911 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +17 |
| 2026-03-20 | `03-20_11L_XSA4_EMA_Int6_MLP3x_WD04_1.1271` | 1.1271 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +22 |
| 2026-03-20 | `03-20_11L_EfficientPartialXSA_FA3_SWA120` | 2.7543 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +22 |
| 2026-03-20 | `03-20_10L_Int5MLP_MuonWD04_SWA50` | 1.6487 | Adam, AdamW, BigramHash, Cosine Warmdown, EMA +17 |
| 2026-03-19 | `03-19_WarmdownQuantization` | 2.8194 | Adam, Cosine Warmdown, EMA, FP16 Embed, GQA +13 |
| 2026-03-19 | `03-19_TrainingOptSeq4096` | 2.9266 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-19 | `03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit` | None | Adam, AdamW, Cosine Warmdown, EMA, GQA +14 |
| 2026-03-19 | `03-19_SlidingWindowEval` | 2.9048 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +13 |
| 2026-03-19 | `03-19_Seq2048_FP16Emb_TunedLR` | 2.8223 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +13 |
| 2026-03-19 | `03-19_MixedQuant_Int6Int8_SlidingWindow` | 2.8330 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +13 |
| 2026-03-19 | `03-19_MLP3x_QAT_Int6_SlidingWindow` | None | Adam, AdamW, Cosine Warmdown, EMA, FP16 Embed +18 |
| 2026-03-19 | `03-19_10L_MixedPrecision` | 2.9590 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-18 | `03-18_LowerLR` | 2.9791 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-18 | `03-18_LongContextSeq2048` | 2.9372 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-18 | `03-18_FP16Embed_WD3600` | 2.9712 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-17 | `03-17_NaiveBaseline` | 2.9903 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +10 |
| 2026-03-17 | `03-17_LoRA_TTT` | 2.9059 | Adam, Cosine Warmdown, EMA, GQA, Gradient Clipping +13 |
