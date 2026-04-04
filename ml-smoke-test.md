# ML Smoke Test Skill

When asked to test a machine learning change, always:
1. First create a tiny model config (dim=128, layers=2, 300 steps, batch=4096)
2. Check for NaN loss every 50 steps — STOP and report if NaN appears
3. Compare to a baseline run at identical config
4. Report both val_bpb numbers and the delta
5. Save results as JSON to ./results/{test_name}.json
6. Print PASS/FAIL with specific reason
Never run full-size models for smoke tests. Time limit is 15 minutes per test.
