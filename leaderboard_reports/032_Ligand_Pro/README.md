### OpenADMET CYP Challenge: Method Report


## Overview

Our final solution is a fixed, isoform-specific ensemble rather than one shared predictor. The main models are Chemprop message-passing neural networks initialized from the CheMeleon foundation model. They are complemented by isoform specialists, RDKit 2D CatBoost models, and small Uni-Mol branches. Each CYP isoform has its own fixed blend because the useful auxiliary data and model diversity differed substantially between targets.

## Data preprocessing

The official challenge data were the primary training source. Public external measurements were used only in selected union, specialist, auxiliary, or pretraining branches.

| Data source | Labels used | Role |
| --- | --- | --- |
| [OpenADMET challenge dataset](https://huggingface.co/datasets/openadmet/cyp-challenge-train-test) | Direct-inhibition pIC50, confidence intervals, measurement variability, and single-concentration screening endpoints | Primary supervision and validation |
| [ChEMBL](https://www.ebi.ac.uk/chembl/) | Curated CYP activity and pIC50/pActivity measurements | Auxiliary tasks and isoform-specific external training |
| [PubChem](https://pubchem.ncbi.nlm.nih.gov/) | Public CYP assay activities | External pretraining and auxiliary supervision for selected branches

1. Molecules were parsed and canonicalized with RDKit. Identical canonical structures were assigned to the same validation partition.
2. Activity values were converted to a common pActivity/pIC50 scale where appropriate. Repeated external measurements for the same canonical structure and task were aggregated robustly, while measurements from different sources could remain separate auxiliary tasks.
3. External data matching a validation or test structure were removed from that fold's external training subset.
4. Model selection used five-fold out-of-fold predictions. The main specialist experiments used ECFP4/Butina structure-cluster splits to test transfer to chemically distinct compounds.
5. Chemprop specialists were averaged across random seeds. Uni-Mol predictions were averaged across fold models before entering the low-weight blends.
6. Blend weights were fixed using out-of-fold results and the challenge scoring implementation. We preferred conservative weights that improved several folds rather than a weight selected from a single split.
7. Final outputs were expressed on the pIC50 scale and clipped to a minimum value of 1.0.

## Models

Prediction target | Final model composition |
| --- | --- |
| CYP1A2 | CheMeleon-initialized graph anchor, graph model trained on the union dataset, RDKit/CatBoost auxiliary model, and two generations of CYP1A2 specialist ensembles |
| CYP2C9 | 90% established graph/tabular ensemble + 5% Uni-Mol direct branch + 5% Uni-Mol transfer branch |
| CYP2D6 | 85% established ensemble + 7.5% first Uni-Mol transfer ensemble + 7.5% independently seeded Uni-Mol transfer ensemble |
| CYP3A4 | 75% established CYP3A4 ensemble + 25% multi-head Chemprop/CheMeleon model

The ensemble contained the following model families:

- **Chemprop + CheMeleon anchor:** a multitask graph neural network initialized from CheMeleon.
- **Chemprop union:** a related graph model trained on the controlled union dataset.
- **Isoform specialists:** Chemprop models trained separately for individual CYP isoforms and averaged across seeds.
- **RDKit 2D CatBoost:** CatBoost regression on RDKit 2D descriptors, using 1,800 trees, depth 8, and MAE loss.
- **Uni-Mol branches:** five-fold molecular representation models used at low weights for complementary predictions.

The final CYP1A2 prediction used the graph anchor, union model, RDKit/CatBoost auxiliary model, and two generations of CYP1A2 specialists. The original blend was 16% anchor, 48% union, 16% CatBoost, and 20% specialist; this prediction was subsequently combined with an extended specialist ensemble selected by out-of-fold validation.

For CYP2C9, the established graph/tabular ensemble contributed 90% of the final prediction. Uni-Mol direct and transfer branches contributed 5% each.

For CYP2D6, 85% came from the established ensemble and 7.5% from each of two independently seeded Uni-Mol transfer branches.

For CYP3A4, the established prediction combined the CheMeleon anchor, union model, standard CYP3A4 specialists, an auxiliary-heavy specialist, and a 15% Uni-Mol branch. The final submission then replaced 25% of this CYP3A4 prediction with a multi-head Chemprop + CheMeleon model.

The CYP3A4 multi-head model jointly predicted:

1. direct-inhibition pIC50;
2. interval center and interval half-width;
3. measurement standard deviation;
4. ChEMBL CYP3A4 pIC50;
5. an auxiliary CYP3A4 screening response.

It used MAE loss, two 600-unit feed-forward layers, dropout 0.1, early stopping, and multiple random seeds. Its point prediction was constrained to its predicted interval before blending. No blind-set value required interval clipping in the submitted file.

## Validation and ensemble selection

Before submission, we checked the CSV using the official activity-submission validator and independently verified:

- 750 prediction rows;
- the required molecule identifiers and column order;
- finite numeric predictions for all four isoforms;
- preservation of columns not modified by the final CYP3A4 blend.

The final 25% multi-head update changed only the CYP3A4 column relative to the preceding accepted ensemble. CYP1A2, CYP2C9, and CYP2D6 were preserved.
