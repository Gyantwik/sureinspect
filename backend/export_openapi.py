"""
backend/export_openapi.py
Exports the OpenAPI specification to contracts/openapi.json.
Used by CI to prevent schema drift and unblock frontend type generation.
"""
import json
from pathlib import Path
from backend.app_factory import create_app

def export():
    app = create_app()
    openapi_schema = app.openapi()
    contracts_dir = Path(__file__).resolve().parents[1] / "contracts"
    contracts_dir.mkdir(parents=True, exist_ok=True)
    out_file = contracts_dir / "openapi.json"
    out_file.write_text(json.dumps(openapi_schema, indent=2), encoding="utf-8")
    print(f"Exported OpenAPI specification to {out_file}")

if __name__ == "__main__":
    export()
