import json
import traceback
from pathlib import Path

nb_path = Path('/Users/panshulaj/Documents/sale forecasting/walmart_sales_forecasting.ipynb')
nb = json.loads(nb_path.read_text(encoding='utf-8'))

ctx = {'__name__': '__main__'}

for idx, cell in enumerate(nb.get('cells', []), start=1):
    if cell.get('cell_type') != 'code':
        continue
    code = ''.join(cell.get('source', []))
    print(f"\n--- Executing code cell {idx} ---")
    try:
        exec(compile(code, f"<cell {idx}>", 'exec'), ctx, ctx)
    except Exception:
        print(f"Cell {idx} FAILED")
        traceback.print_exc()
        raise SystemExit(1)

print("\nAll code cells executed successfully.")
