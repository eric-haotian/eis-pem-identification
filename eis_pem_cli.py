import sys
import json
import argparse
from eis_pem.frontend import identify_parameters_robust

def main():
    parser = argparse.ArgumentParser(description="EIS-PEM Parameter Identification CLI Tool")
    parser.add_argument("input_json", help="Path to the input JSON configuration file")
    parser.add_argument("output_json", help="Path to save the output JSON results")
    args = parser.parse_args()

    try:
        with open(args.input_json, 'r') as f:
            request = json.load(f)
    except Exception as e:
        print(f"Error reading input JSON file: {e}")
        sys.exit(1)

    print("Running EIS-PEM Identification pipeline...")
    try:
        # Run the full pipeline
        result = identify_parameters_robust(request)
    except Exception as e:
        print(f"Algorithm failed: {e}")
        sys.exit(1)

    try:
        with open(args.output_json, 'w') as f:
            json.dump(result, f, indent=4)
        print(f"Success! Results saved to {args.output_json}")
    except Exception as e:
        print(f"Error saving output JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
