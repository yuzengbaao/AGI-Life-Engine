def main():
    with open('data\insights\insight_1766721839.md', 'r') as f:
        content = f.read()
    summary = 'Summary of insight: ' + content[:min(len(content), 200)] + '... (truncated)'