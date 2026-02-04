import sys
import os
import json

def analyze_experiment_output(file_path):
    """
    Parses the experiment output file and returns a JSON summary.
    Designed to be robust against different output formats.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)

    result = {
        "high_entropy": None,
        "low_entropy": None,
        "threshold": None,
        "creation_required": False,
        "raw_content": ""
    }

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            result["raw_content"] = content
            
            # Parse line by line
            for line in content.splitlines():
                line = line.strip()
                if not line: continue
                
                if "High Entropy:" in line:
                    try:
                        result["high_entropy"] = float(line.split(":")[1].strip())
                    except: pass
                elif "Low Entropy:" in line:
                    try:
                        result["low_entropy"] = float(line.split(":")[1].strip())
                    except: pass
                elif "Threshold:" in line:
                    try:
                        result["threshold"] = float(line.split(":")[1].strip())
                    except: pass
                elif "Creation Required:" in line:
                    val = line.split(":")[1].strip().lower()
                    result["creation_required"] = val == "true"

        print(json.dumps(result, indent=2))
        return 0

    except Exception as e:
        print(f"Error analyzing data: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Default to sandbox output if no argument provided
    target_file = sys.argv[1] if len(sys.argv) > 1 else r"D:\TRAE_PROJECT\AGI\data\sandbox\experiment_output.txt"
    sys.exit(analyze_experiment_output(target_file))
