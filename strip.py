import tokenize
import io
import sys

def remove_comments(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        source = f.read()
    
    io_obj = io.StringIO(source)
    out = ""
    last_lineno = -1
    last_col = 0
    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type = tok[0]
        token_string = tok[1]
        start_line, start_col = tok[2]
        end_line, end_col = tok[3]
        
        # Format padding
        if start_line > last_lineno:
            last_col = 0
        if start_col > last_col:
            out += (" " * (start_col - last_col))
            
        if token_type == tokenize.COMMENT:
            pass
        else:
            out += token_string
            
        last_lineno = end_line
        last_col = end_col

    # Since dropping full-line comments leaves empty lines with spaces, let's clean up
    cleaned_lines = []
    for line in out.splitlines():
        if line.strip() == "" and line != "":
            # keep empty line but no spaces
            cleaned_lines.append("")
        else:
            cleaned_lines.append(line)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("\n".join(cleaned_lines) + "\n")

remove_comments("records/track_non_record_16mb/2026-04-11_8L_GQA_PartialRoPE_Int8_AttnMLP3_QAT015/train_gpt.py")
