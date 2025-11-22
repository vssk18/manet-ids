#!/bin/bash

# ==============================================================================
# MANET IDS Paper 1 - GitHub Upload Script
# ==============================================================================

echo "=================================================="
echo "Paper 1: MANET IDS - GitHub Upload"
echo "=================================================="
echo ""

# Step 1: Navigate to your repository
echo "Step 1: Navigate to your manet-ids repository"
echo "Run: cd /path/to/manet-ids"
echo "Press Enter when ready..."
read

# Step 2: Create paper directory structure
echo ""
echo "Step 2: Creating paper directory structure..."
mkdir -p paper/figures
echo "✓ Created paper/figures/"

# Step 3: Copy files (you'll need to update paths)
echo ""
echo "Step 3: Copy the following files to your repo:"
echo ""
echo "From the downloaded ZIP, copy:"
echo "  • generate_figures.py → paper/"
echo "  • All .pdf files from figures/ → paper/figures/"
echo "  • SUBMISSION_GUIDE.md → paper/"
echo "  • main.tex (your LaTeX file) → paper/"
echo ""
echo "Press Enter when files are copied..."
read

# Step 4: Stage changes
echo ""
echo "Step 4: Staging changes for commit..."
git add paper/

# Step 5: Commit
echo ""
echo "Step 5: Committing changes..."
git commit -m "Add research paper for arXiv and Ad Hoc Networks submission

- Complete LaTeX paper (main.tex) with 7 publication figures
- XGBoost achieves 94.7% multiclass, 96.7% binary accuracy
- ROC-AUC: 0.994
- Leakage-free evaluation with 5-fold stratified CV
- Feature importance: Queue Length, Buffer Util, Fwd Consistency
- Python script to generate all figures from experimental data
- Submission guide for arXiv and journal

Ready for arXiv submission and journal review."

# Step 6: Push to GitHub
echo ""
echo "Step 6: Pushing to GitHub..."
git push origin main

echo ""
echo "=================================================="
echo "✅ Upload Complete!"
echo "=================================================="
echo ""
echo "Next Steps:"
echo "1. Verify files at: https://github.com/vssk18/manet-ids"
echo "2. Create paper/ folder should be visible"
echo "3. Check paper/figures/ has all 7 PDFs"
echo "4. Ready for arXiv submission!"
echo ""
echo "GitHub Repository Structure:"
echo "manet-ids/"
echo "├── data/"
echo "│   └── manet_dataset.csv"
echo "├── src/"
echo "│   └── (your Python code)"
echo "├── models/"
echo "│   └── (trained models)"
echo "├── paper/                    ← NEW"
echo "│   ├── main.tex              ← LaTeX source"
echo "│   ├── SUBMISSION_GUIDE.md   ← Instructions"
echo "│   ├── generate_figures.py   ← Figure generator"
echo "│   └── figures/              ← All 7 PDFs"
echo "│       ├── fig_01_class_distribution.pdf"
echo "│       ├── fig_02_correlation.pdf"
echo "│       ├── fig_03_roc_curves.pdf"
echo "│       ├── fig_04_confusion_matrix.pdf"
echo "│       ├── fig_05_feature_importance.pdf"
echo "│       ├── fig_06_distributions.pdf"
echo "│       └── fig_07_model_comparison.pdf"
echo "└── README.md"
echo ""

# ==============================================================================
# ALTERNATIVE: Quick Commands (if you know what you're doing)
# ==============================================================================

cat << 'EOF_QUICK'

QUICK COMMANDS (copy-paste all at once):
-----------------------------------------

cd ~/path/to/manet-ids
mkdir -p paper/figures
# Copy files here (from your Downloads/Paper1_MANET_IDS_Submission.zip)
git add paper/
git commit -m "Add research paper with 7 publication figures

XGBoost: 94.7% multiclass, 96.7% binary accuracy, ROC-AUC 0.994
Ready for arXiv and Ad Hoc Networks submission"
git push origin main

EOF_QUICK

echo "=================================================="
