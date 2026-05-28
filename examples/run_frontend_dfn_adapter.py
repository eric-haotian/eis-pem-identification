"""Example JSON-style frontend request routed into the DFN forward model."""

from pathlib import Path
import json

from eis_pem import dfn_frontend_response_to_frame, simulate_dfn_from_frontend

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    request = {
        "frequency": {"min_hz": 1e-2, "max_hz": 1e5, "points": 80},
        "conditions": [
            {"temperature_K": 298.15, "SOC": 0.50},
        ],
        "response_channels": ["cell", "neg", "pos", "sep"],
        "parameters": {
            "R_contact": 0.01,
            "L_ind": 1e-8,
        },
    }
    response = simulate_dfn_from_frontend(request)
    frame = dfn_frontend_response_to_frame(response)

    data_dir = PROJECT_ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    response_path = data_dir / "frontend_dfn_response.json"
    table_path = data_dir / "frontend_dfn_spectrum.csv"

    response_path.write_text(json.dumps(response, indent=2), encoding="utf-8")
    frame.to_csv(table_path, index=False)

    print("Frontend DFN adapter example")
    print(f"Model: {response['model']}")
    print(f"Parameters exposed: {response['parameter_count']}")
    print(f"Spectrum rows: {len(response['spectra'])}")
    print(f"Response JSON written: {response_path}")
    print(f"Spectrum CSV written:  {table_path}")


if __name__ == "__main__":
    main()
