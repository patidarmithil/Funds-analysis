import os
pages = [
    'app_pages/overview.py',
    'app_pages/analysis.py', 
    'app_pages/predictions.py',
    'app_pages/risk_analysis.py',
    'app_pages/backtesting.py',
    'app_pages/simulation.py',
    'app_pages/manual.py',
]
for p in pages:
    if os.path.exists(p):
        with open(p, 'r', encoding='utf-8') as f:
            content = f.read()
        content = content.replace('use_container_width=True', "width='stretch'")
        with open(p, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Updated: {p}")
print("Done!")
