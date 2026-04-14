import os

log_file = '/Users/akhileshgogikar/parameter-golf/logs/remote_run_20260413/clean_log_mac.txt'

if os.path.exists(log_file):
    with open(log_file, 'rb') as f:
        data = f.read()
    
    # Split by both \n and \r
    lines = data.replace(b'\r', b'\n').split(b'\n')
    print(f"Total lines after replacement: {len(lines)}")
    
    # Print the last 50 lines that aren't empty
    printed = 0
    for line in reversed(lines):
        if line.strip():
            print(line.decode('utf-8', errors='ignore'))
            printed += 1
        if printed >= 50:
            break
else:
    print("File not found.")
