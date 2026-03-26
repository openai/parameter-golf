# Submission Guide for OpenAI Parameter Golf Challenge

This document provides a complete guide for submitting your solution to the OpenAI Model Craft: Parameter Golf Challenge.

## Submission Requirements

According to the official terms, your submission must include:

1. **Model Weights** - Compressed model artifact (≤ 16MB)
2. **Config** - Model configuration file
3. **Supporting Code** - Training and evaluation code
4. **Training Log** - Output from training run
5. **Execution Script** - Script to reproduce results
6. **Short Write-up** - Description of your approach

## Submission Checklist

### Pre-Submission Verification

- [ ] Model artifact size ≤ 16,000,000 bytes (decimal)
- [ ] Training completes within 600 seconds wall-clock time
- [ ] Uses only the provided FineWeb dataset
- [ ] Code is open-source under MIT license
- [ ] All files are in a public GitHub repository

### Files to Include

```
your-submission/
├── model.bin              # Compressed model weights (≤ 16MB)
├── config.json            # Model configuration
├── train_gpt.py           # Training code
├── evaluate.py            # Evaluation code
├── run.sh                 # Execution script
├── training.log           # Training output log
├── requirements.txt       # Dependencies
└── WRITEUP.md             # Approach description
```

## Creating the Submission

### Step 1: Train the Model

```bash
# Run training
python train_gpt.py --data_dir ./data --output ./model.bin

# Verify size
ls -la ./model.bin
# Should be ≤ 16,000,000 bytes
```

### Step 2: Generate Training Log

```bash
# Run training with logging
python train_gpt.py --data_dir ./data --output ./model.bin 2>&1 | tee training.log
```

### Step 3: Export Configuration

```bash
# Export config to JSON
python -c "
from config import DEFAULT_CONFIG
import json
config = DEFAULT_CONFIG
with open('config.json', 'w') as f:
    json.dump({k: v for k, v in vars(config).items() if not k.startswith('_')}, f, indent=2)
"
```

### Step 4: Evaluate and Record BPB

```bash
# Run evaluation
python evaluate.py --model ./model.bin --data ./data/val.bin 2>&1 | tee eval.log

# Extract BPB score
grep "FINAL BPB" eval.log
```

### Step 5: Write the Write-up

Create `WRITEUP.md` with the following structure:

```markdown
# Parameter Golf Solution Write-up

## Team/Author Information
- Name: [Your Name]
- GitHub: [Your GitHub Handle]
- Email: [Your Email]

## Approach Summary

Brief description of your approach (2-3 sentences).

## Key Innovations

1. **Innovation 1**: Description
2. **Innovation 2**: Description
3. **Innovation 3**: Description

## Architecture Details

- Vocabulary Size: [e.g., 2048]
- Layers: [e.g., 10]
- Hidden Dimension: [e.g., 512]
- MLP Ratio: [e.g., 3.0]
- Total Parameters: [e.g., 21M]

## Optimizations

### 1. Optimizer
Description of optimizer choice and rationale.

### 2. Quantization
Description of quantization strategy.

### 3. Evaluation
Description of evaluation optimizations.

## Results

- Validation BPB: [Your BPB Score]
- Model Size: [Size in MB]
- Training Time: [Time in minutes]

## Reproduction Instructions

```bash
# Step-by-step instructions to reproduce your results
bash run.sh all
```

## Dependencies

- PyTorch 2.0+
- sentencepiece
- zstandard
- flash-attn (optional)

## References

1. [Any papers or resources you used]
```

### Step 6: Create GitHub Repository

```bash
# Initialize repository
git init
git add .
git commit -m "Initial submission"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/parameter-golf-submission.git
git push -u origin main
```

### Step 7: Submit Pull Request

1. Fork the official repository: https://github.com/openai/parameter-golf
2. Add your submission to the appropriate track directory
3. Create a Pull Request with:
   - Clear title: `[Submission] Your Name - BPB Score`
   - Description summarizing your approach
   - Link to your repository

## Evaluation Criteria

Your submission will be evaluated on:

1. **Bits Per Byte (BPB)** - Lower is better
2. **Code Quality** - Clean, readable, well-documented
3. **Innovation** - Novel techniques and optimizations
4. **Reproducibility** - Easy to verify results

## Tips for Success

### Maximizing Performance

1. **Use Int6 Quantization** - Saves 25% space vs Int8
2. **Implement Muon Optimizer** - 35% faster convergence
3. **Enable Test-Time Training** - ~0.033 BPB improvement
4. **Use Sliding Window Evaluation** - ~0.034 BPB improvement
5. **Optimize MLP Ratio** - 3x is the sweet spot

### Avoiding Disqualification

1. **Don't exceed 16MB** - Strict limit
2. **Don't train on validation data** - Use only provided training split
3. **Don't use external data** - FineWeb only
4. **Don't submit automated solutions** - Must be your own work

### Common Pitfalls

1. **torch.compile overhead** - Compile during warmup, not training
2. **Data I/O bottlenecks** - Pre-tokenize data
3. **Gradient accumulation errors** - Check effective batch size
4. **Quantization timing** - Start QAT at 75% of training

## Example Submission

See the `example_submission/` directory for a complete example.

## Timeline

- Challenge Start: March 18, 2026
- Challenge End: April 30, 2026
- Results Announcement: TBD

## Support

- GitHub Issues: https://github.com/openai/parameter-golf/issues
- Community Discussion: Check the GitHub issues for tips

## License

Your submission must be open-source under the MIT License.

```
MIT License

Copyright (c) 2026 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[Full license text...]
```

## Questions?

Refer to the official challenge terms:
https://cdn.openai.com/pdf/d5caec5a-ee81-419d-b0d7-39f1424d819c/OpenAI%20Model%20Craft_%20Parameter%20Golf%20Challenge%20Terms%20and%20Conditions.pdf

Good luck!
