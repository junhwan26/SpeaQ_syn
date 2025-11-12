#!/bin/bash

# Label Distribution Comparison Analysis Runner
# This script runs the label distribution comparison analysis

echo "=== SpeaQ Label Distribution Analysis ==="
echo "Note: Please make sure you have activated the speaq_analysis conda environment"
echo "You can activate it with: conda activate speaq_analysis"
echo ""

# Check if we're in the correct conda environment
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "Warning: No conda environment detected."
    echo "Please activate the speaq_analysis environment first:"
    echo "  conda activate speaq_analysis"
    echo ""
    echo "Or run the Python script directly:"
    echo "  python label_distribution_comparison.py --help"
    exit 1
elif [ "$CONDA_DEFAULT_ENV" != "speaq_analysis" ]; then
    echo "Warning: Current environment is '$CONDA_DEFAULT_ENV', but 'speaq_analysis' is expected."
    echo "Please activate the speaq_analysis environment:"
    echo "  conda activate speaq_analysis"
    exit 1
fi

echo "✓ Conda environment detected: $CONDA_DEFAULT_ENV"

# Set default parameters
CONFIG_FILE="/home/junhwanheo/SpeaQ/configs/speaq_multi_dataset.yaml"
OUTPUT_DIR="label_distribution_analysis"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--config-file PATH] [--output-dir DIR]"
            echo ""
            echo "Options:"
            echo "  --config-file PATH    Path to config YAML file (default: $CONFIG_FILE)"
            echo "  --output-dir DIR      Output directory for results (default: $OUTPUT_DIR)"
            echo "  --help               Show this help message"
            echo ""
            echo "Note: Make sure to activate the speaq_analysis conda environment first:"
            echo "  conda activate speaq_analysis"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Running label distribution analysis..."
echo ""

# Run the analysis
python /home/junhwanheo/SpeaQ/analysis/label_distribution_comparison.py \
    --config-file "$CONFIG_FILE" \
    --output-dir "$OUTPUT_DIR"

if [ $? -eq 0 ]; then
    echo ""
    echo "=== Analysis Complete ==="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "  - object_class_distribution.png"
    echo "  - predicate_distribution.png"
    echo "  - top_classes_comparison.png"
    echo "  - top_predicates_comparison.png"
    echo "  - distribution_statistics.png"
    echo "  - label_distribution_report.txt"
    echo ""
    echo "You can view the results in the $OUTPUT_DIR directory."
else
    echo ""
    echo "Error: Analysis failed. Please check the error messages above."
    exit 1
fi