# Agent Change Log — JEPA v2

## 2026-04-01 — Sessione 1: Diagnosi v1 + implementazione stack completo

### Diagnosi: perché JEPA v1 non funzionava

Dall'analisi dei log della v1 (c4e8f9a5 vs a9ea3137):

| Metrica | JEPA ON | JEPA OFF |
|---------|---------|----------|
| Steps in 600s | 430 | 693 |
| Step avg | 1396ms | 867ms |
| val_bpb finale | 1.6153 | 1.4783 |
| A step 400 (equo) | 1.6132 | 1.5861 |

**3 bug identificati:**

1. **EMA momentum troppo alto (0.996)**: con 430 step, il target encoder ha aggiornato
   solo `430 × (1 - 0.996) = 1.72%` dei suoi pesi → target ≈ online encoder → task triviale.
   Il predictor parte con `_zero_init=True` (output=0), quindi `z_pred = z_context` inizialmente.
   L'EMA target è quasi uguale al base model → `MSE(norm(z), norm(z)) ≈ 0` al primo step.
   Risultato: `jepa_loss = 0.002` costante, gradiente quasi zero, nessun segnale.

2. **Single-step prediction troppo facile**: predire `z[t+1]` da `z[t]` è quasi ridondante
   con il CE objective che già forza `z[t]` a contenere info su `t+1`.

3. **Bug grad accumulation**: `z_target_cached` calcolato da `micro_batches[0]`, poi
   applicato come target a TUTTI i micro-step (che usano batch diversi). 7/8 micro-step
   hanno JEPA loss su coppie (prediction_batch_B, target_batch_A) → rumore puro.

### Fix implementati

#### Fix 1: EMA momentum 0.9 (era 0.996)
Con 0.9, dopo 50 step il target ha ricevuto `50 × 0.1 = 5` unità di aggiornamento → diverge
abbastanza da rendere il task non-triviale. Mezzo-vita ≈ 7 step.

#### Fix 2: Multi-step prediction [1, 2, 4, 8]
Un singolo `encode(x)` del target encoder, poi loss calcolata a 4 offset:
```python
for offset, w in [(1, 1.0), (2, 0.5), (4, 0.25), (8, 0.125)]:
    z_p = z_pred[:, :T-offset, :]
    z_t = z_target_full[:, offset:, :]
    jepa_ms_loss += w * MSE(norm(z_p), norm(z_t))
jepa_loss = jepa_ms_loss / total_weight  # normalizzato
```
Il target encoder gira UNA VOLTA, le 4 loss sono semplici slicing. Overhead minimo.
Offset-8 richiede pianificazione a lungo raggio → non triviale nemmeno per un causal LM.

#### Fix 3: z_target per micro-batch corretto
Il target encoder ora gira dentro il loop dei micro-step, sulla stessa `x` del forward CE.
Overhead: `grad_accum_steps × encode_time` invece di `1 × encode_time`.
Trade-off documentato: +overhead ma gradienti corretti su tutti gli 8 micro-step.

### Nuove features

#### BigramHash(2048)
Lookup table per coppie (token[t-1], token[t]) via hash di Cantor:
```
h(a, b) = (a+b)(a+b+1)//2 + b  mod 2048
```
Output sommato all'embedding prima del primo layer. ~1.05M params, ~300-500KB compresso.
Libera capacità attention da statistica bigrammica.

#### int6 + LZMA
- `QUANT_MAX=31`: range quantizzazione [-31, 31] in container int8
- `lzma.compress(preset=9)` invece di `zlib.compress(level=9)`
- Risparmio atteso: ~2-3 MB artifact + ~280 KB rispetto a zlib

#### Artifact EMA (decay=0.9999)
Distinto dall'EMA JEPA (target encoder). Questo è una media Polyak dei pesi durante
il training, salvata come checkpoint finale. Aggiornato ogni step dopo `optimizer.step()`.

#### LeakyReLU(0.5)²
`F.leaky_relu(x, 0.5).square()` invece di `relu(x).square()`. Community-validated,
free -0.001 BPB circa. Cambia 1 riga nel MLP.

### Ablation modes in run.sh
- `full`: stack completo
- `ablation`: solo CE, no JEPA, no BigramHash (baseline di confronto)
- `jepa_only`: JEPA senza BigramHash (isola contributo JEPA)
- `bigram_only`: BigramHash senza JEPA (isola contributo BigramHash)
- `smoke`: 2 min, tutto attivo

### TODO pre-PR
- [ ] Run smoke test per verificare che giri senza errori
- [ ] Run `full` + `ablation` per confronto fair (stesso wallclock)
- [ ] Verificare che artifact sia < 16 MB
- [ ] Misurare jepa_loss effettiva (deve essere >> 0.002 con momentum=0.9)
- [ ] Scrivere README con risultati reali
