# HIP Platform — Hardware Data Parsers

> Complete documentation for all parser files.  
> Built for the HIP (Hardware Intelligence Platform) project.

---

## Quick Start

Put all 4 files in the same folder, then double-click `RUN_PARSERS.bat`.

```
your_folder/
  ├── RUN_PARSERS.bat                ← double-click this
  ├── cmsis_parser_universal.py
  ├── modm_unified_parser.py
  └── modm_batch_runner.py
```

Output: `all_modm_data.json` in your Downloads folder.

---

## What Each File Does

### `RUN_PARSERS.bat`
**The only file you need to touch.**  
Windows batch script that does everything automatically:
1. Clones the 3 GitHub repos (first run only)
2. Installs Python dependencies
3. Generates modm device data (first run only, takes 2-5 min)
4. Runs all 3 parsers
5. Saves output to your Downloads folder

Works on any Windows PC with any username — no hardcoded paths.

---

### `cmsis_parser_universal.py`
**Parses CMSIS device header files (`.h`)**

**Works with any CMSIS header — not just modm.**  
You can point it at:
- Headers from `modm-io/cmsis-header-stm32` (what batch runner uses)
- Headers downloaded directly from ST's website
- Headers from your STM32CubeIDE installation
- Headers from any other source — it doesn't care

**What it extracts:**

| Data | Example |
|------|---------|
| Base addresses | `USART1 → 0x40011000, bus: APB2` |
| Register offsets | `CR1 → offset: 0x0C, access: RW` |
| Register descriptions | `"USART Control register 1"` |
| Bit field positions | `USART_CR1_UE → pos: 13, width: 1` |
| Bit field descriptions | `"USART Enable"` |
| IRQ numbers | `USART1_IRQn → 37` |
| RCC clock map | `USART1 → APB2ENR, bit 4` |
| Memory regions | `SRAM1 → 0x20000000, 112KB, DMA OK` |
| Reserved members | `RESERVED0[88] → 0x020-0x17F` |
| Core feature flags | `FPU_PRESENT=1, NVIC_PRIO_BITS=4` |
| System info | `SystemCoreClock, SystemInit()` |

**CLI usage:**
```bash
# Single file — auto-detects target device from filename
python cmsis_parser_universal.py --file stm32f407xx.h

# Whole folder
python cmsis_parser_universal.py --folder cmsis-header-stm32/stm32f4xx/Include/

# With explicit target
python cmsis_parser_universal.py --file stm32f407xx.h --target STM32F407xx

# Custom output path
python cmsis_parser_universal.py --file stm32f407xx.h --output my_output.json
```

**Auto-detection:** The parser figures out the target device automatically from:
1. `#ifndef` include guard (`__STM32F407xx_H`) — score 95/100
2. `@file` comment — score 92/100
3. Filename (`stm32f407xx.h`) — score 90/100
4. Family fallback (`system_stm32f4xx.h` → STM32F4) — score 50/100

**Also has a GUI:**
```bash
python cmsis_parser_ui.py
```
Simple drag-and-drop window, auto-detects device, no typing needed.

---

### `modm_unified_parser.py`
**Parses modm-devices XML files and modm-data SVD files**

Two sources in one file:

**Source 1 — modm-devices XML** (stable, use this)
```bash
python modm_unified_parser.py \
    --devices-folder modm-devices/devices/stm32/ \
    --output modm_output.json
```

| Data | Example |
|------|---------|
| GPIO AF matrix | `PA9 → USART1_TX at AF7` |
| DMA assignments | `DMA2 Stream2 Ch4 → SPI1_RX` |
| Memory regions | `CCM RAM → 0x10000000, 64KB, NO DMA` |
| Peripheral list | `USART: 1,2,3,6 / SPI: 1,2,3` |
| Device variants | `STM32F407VGT6: LQFP100, 1MB flash` |
| EXTI line map | `EXTI0 shared by PA0, PB0, PC0` |

**Source 2 — modm-data SVD** (beta, use as validation only)
```bash
python modm_unified_parser.py \
    --svd-folder modm-data/ext/stm32/svd/ \
    --output modm_output.json
```

| Data | Example |
|------|---------|
| Peripheral base addresses | `USART1 → 0x40011000` |
| Register maps | `CR1 offset 0x0C` |
| Bit fields | `UE → pos 13, width 1` |
| Reset values | `CR1 reset: 0x00000000` |
| Field enum values | `STOP: 00=1bit, 10=2bits` |
| IRQ numbers | `USART1 → 37` |

**With cross-validation:**
```bash
python modm_unified_parser.py \
    --devices-folder modm-devices/devices/stm32/ \
    --svd-folder modm-data/ext/stm32/svd/ \
    --validate-cmsis cmsis_output.json \
    --output modm_output.json
```

---

### `modm_batch_runner.py`
**Runs everything at once across all 3 repos**

This is what `RUN_PARSERS.bat` calls internally.  
You can also run it directly from command line:

```bash
# Auto-detect all repos in current folder
python modm_batch_runner.py --all

# Explicit paths
python modm_batch_runner.py \
    --cmsis-headers cmsis-header-stm32/ \
    --svd           cmsis-svd-stm32/ \
    --devices       modm-devices/devices/stm32/ \
    --output        all_data.json

# One family only (good for testing — much faster)
python modm_batch_runner.py \
    --cmsis-headers cmsis-header-stm32/ \
    --svd           cmsis-svd-stm32/ \
    --family        stm32f4 \
    --output        f4_only.json
```

**Important:** Each device header file gets its own auto-detected target.  
`stm32f407xx.h` and `stm32f446xx.h` are in the same folder but are different chips — the batch runner handles this correctly. Running the CMSIS parser on the whole folder with one global target would give wrong data.

---

## The 3 GitHub Repos

### `modm-io/cmsis-header-stm32`
CMSIS device headers for all STM32 families.  
Structure: `stm32f4xx/Include/stm32f407xx.h`  
369 header files, 24 families (F0, F1, F2, F3, F4, F7, G0, G4, H5, H7, L0, L1, L4, L5, U5, WB...)  
License: Apache-2.0

### `modm-io/cmsis-svd-stm32`
CMSIS SVD register definition files for all STM32 families.  
Structure: `stm32f4/STM32F407.svd`  
192 SVD files, 26 families  
License: Apache-2.0

### `modm-io/modm-devices`
Hardware topology data — GPIO alternate functions, DMA routing, memory.  
Structure: `devices/stm32/*.xml` (generated, run `make generate-stm32` first)  
110 XML files covering all STM32 families  
License: MPLv2

---

## Output JSON Structure

```
all_modm_data.json
│
├── metadata
│     ├── sources           ← stats for each source
│     └── cross_validation  ← discrepancy counts
│
├── cmsis_data              ← from cmsis-header-stm32
│     ├── base_address_macros      ← peripheral addresses + bus
│     ├── peripheral_structs       ← registers + offsets + descriptions
│     ├── bit_field_definitions    ← positions + widths + descriptions
│     ├── irq_table                ← IRQ numbers + types
│     ├── rcc_clock_map            ← peripheral → RCC register → bit
│     ├── memory_regions           ← SRAM/Flash/CCM sizes + DMA flag
│     └── core_feature_flags       ← FPU, MPU, NVIC bits
│
├── svd_data                ← from cmsis-svd-stm32
│     └── per device:
│           ├── peripherals        ← base addresses
│           ├── registers          ← offsets + reset values
│           ├── fields             ← bit positions + enum values
│           └── irqs               ← interrupt numbers
│
├── modm_device_data        ← from modm-devices
│     └── per device:
│           ├── gpio_af_matrix     ← (pin, peripheral, signal) → AF number
│           ├── dma_assignments    ← stream/channel → peripheral
│           ├── memory_regions     ← SRAM/Flash sizes + DMA accessible flag
│           ├── exti_line_map      ← which pins share each EXTI line
│           ├── peripherals        ← availability + instance count
│           └── device_variants    ← package/flash/RAM combinations
│
└── cross_validation        ← discrepancies between sources
      ├── ADDRESS_MISMATCH         ← SVD vs CMSIS base address differs
      ├── IRQ_MISMATCH             ← SVD vs CMSIS IRQ number differs
      ├── BIT_POSITION_MISMATCH    ← SVD vs CMSIS bit position differs
      └── BIT_WIDTH_MISMATCH       ← SVD vs CMSIS bit width differs
```

---

## What Data Comes From Where

| Data | CMSIS Headers | SVD Files | modm-devices |
|------|:---:|:---:|:---:|
| Base addresses | ✅ | ✅ | — |
| Register offsets | ✅ | ✅ | — |
| Bit positions/widths | ✅ | ✅ | — |
| Bit descriptions | ✅ | ✅ | — |
| IRQ numbers | ✅ | ✅ | — |
| Reset values | — | ✅ | — |
| Field enum values | — | ✅ | — |
| RCC clock map | ✅ | — | — |
| GPIO AF matrix | — | — | ✅ |
| DMA assignments | — | — | ✅ |
| Memory sizes | ✅ | — | ✅ |
| EXTI line map | — | — | ✅ |
| Package/pin count | — | — | ✅ |

Wherever two sources have the same data (addresses, IRQs, bit positions) the batch runner compares them and flags differences in `cross_validation`.

---

## Coverage

| Source | Families | Files | Devices |
|--------|----------|-------|---------|
| cmsis-header-stm32 | 24 | 369 | 300+ |
| cmsis-svd-stm32 | 26 | 192 | 187 |
| modm-devices | 20+ | 110 XML | 1171+ |

All major STM32 families: F0, F1, F2, F3, F4, F7, G0, G4, H5, H7, L0, L1, L4, L5, U5, WB, WL and more.

---

## How It Feeds HIP

| HIP Layer | Data Used | Source |
|-----------|-----------|--------|
| L1 SVD Engine | Base addresses, registers, bit fields | CMSIS + SVD |
| L3 CrossRef | Discrepancies between sources | cross_validation |
| L4 Validator — CLOCK_COMPAT | RCC clock map | CMSIS |
| L4 Validator — DMA_CONFLICT | DMA assignments + CCM flag | modm-devices |
| L4 Validator — PIN_MUX_CHECK | EXTI line map | modm-devices |
| L5 CodeGen | HAL macros, register values | CMSIS + SVD |
| Solver — AF constraints | GPIO AF matrix | modm-devices |
| Solver — DMA constraints | DMA stream/channel table | modm-devices |

---

## Troubleshooting

**"cmsis_parser_universal.py not found"**  
All 4 `.py` files must be in the same folder as `RUN_PARSERS.bat`.

**Cross-validation shows 0 checks**  
CMSIS parser didn't run — check that `cmsis_parser_universal.py` is in the same folder.

**"modm device data already generated - skipping" but no data**  
Delete `modm-devices/devices/` folder and re-run.

**`make generate-stm32` fails on Windows**  
Run manually:
```bash
cd modm-devices/tools/generator
pip install -r requirements.txt
python generate.py --device stm32
```

**Output JSON is very large (100MB+)**  
Normal — you have data for 300+ devices. Use `--family stm32f4` to limit scope during testing.
