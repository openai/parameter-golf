import re
import json
import os
from datetime import datetime

def extract_leaderboard():
    readme_path = "README.md"
    if not os.path.exists(readme_path):
        return []
        
    with open(readme_path, "r") as f:
        content = f.read()

    # Extract leaderboard table
    table_match = re.search(r"## Leaderboard\n\n(.*?)\n\n", content, re.DOTALL)
    if not table_match:
        return []
        
    table = table_match.group(1)
    rows = [r.strip() for r in table.split("\n") if r.strip()]
    if len(rows) < 3:
        return []
        
    headers = [h.strip() for h in rows[0].split("|")[1:-1]]
    data = []
    for row in rows[2:]:
        cols = [c.strip() for c in row.split("|")[1:-1]]
        if len(cols) == len(headers):
            entry = dict(zip(headers, cols))
            # Clean up score
            try:
                entry["Score"] = float(entry["Score"])
            except:
                pass
            data.append(entry)
    return data

def get_latest_logs():
    logs_path = "logs/diagnostics_20260418"
    if not os.path.exists(logs_path):
        return []
    
    files = [f for f in os.listdir(logs_path) if f.endswith(".txt")]
    logs = []
    for file in files:
        logs.append({
            "name": file,
            "timestamp": datetime.fromtimestamp(os.path.getmtime(os.path.join(logs_path, file))).isoformat()
        })
    return logs

def main():
    data = {
        "leaderboard": extract_leaderboard(),
        "latest_logs": get_latest_logs(),
        "last_updated": datetime.now().isoformat(),
        "project_name": "Parameter Golf",
        "description": "OpenAI Model Craft Challenge: Train the best 16MB model in under 10 minutes."
    }
    
    with open("web/data.json", "w") as f:
        json.dump(data, f, indent=2)
    print("Dashboard data updated in web/data.json")

if __name__ == "__main__":
    main()
