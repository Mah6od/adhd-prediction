# Data

Neither the raw nor the cleaned data are stored in this repository.

1. Download the ADHD-200 Preprocessed phenotypic file (`adhd200_preprocessed_phenotypics.tsv`) from the Preprocessed Connectomes Project (Bellec et al., 2017, https://doi.org/10.1016/j.neuroimage.2016.06.034).
2. Put it in `notebooks/`, next to the notebooks (they read files from their own folder).
3. Run `01_cleaning_multiclass.ipynb`. It writes `adhd_cleaned_multiclass.csv` (505 participants, 11 predictors plus `DX`: 0 control, 1 ADHD-combined, 2 ADHD-inattentive).
4. Run `02_EDA_before_after_cleaning.ipynb`, then `03_ADHD_MAIN_multiclass.ipynb`.
