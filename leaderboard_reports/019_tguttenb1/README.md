# OpenADMET CYP Inhibition Challenge — Modeling Methodology

**Metacoder AI \| CYP1A2, CYP2C9, CYP2D6, and CYP3A4**

## Overview

Our approach integrates complementary molecular representations extracted from the penultimate layers of multiple independently trained graph neural networks (GNNs), message-passing architectures, and three-dimensional molecular models. These models were trained across distinct experimental datasets, endpoints, and CYP isoforms, allowing us to harvest molecular embeddings that encode different aspects of chemical structure, biological activity, and molecular geometry.

Rather than relying exclusively on the predictions of individual models, we systematically recombined these learned embeddings into multiple alternative feature spaces and trained downstream XGBoost (XGB) regression models to predict CYP inhibition. This multirepresentational strategy enables information learned from one experimental endpoint or dataset to contribute to predictions on another, while capturing complementary information from different neural architectures.

We additionally investigated pan-isoform representations, variational autoencoder (VAE) compression, target features, conventional molecular descriptors, and ensemble blending. The final submission combined predictions from independently trained models, XGBoost models operating on harvested embeddings, cross-isoform feature architectures, and externally generated structure-based predictions.

## 1. Datasets

We utilized three datasets provided through the OpenADMET challenge, supplemented by two external sources of CYP inhibition measurements. External data were selectively filtered using generalized molecular scaffold families represented in the challenge test set.

### OpenADMET-Provided Datasets

> **1. Direct CYP inhibition:** The primary challenge training dataset containing quantitative inhibition measurements for CYP1A2, CYP2C9, CYP2D6, and CYP3A4. This dataset served as the principal supervised learning source for the final inhibition predictions.
>
> **2. Single-concentration activity:** An additional OpenADMET dataset containing CYP activity measurements at a single experimental concentration. Models trained on these measurements provided complementary molecular representations that could be transferred to the primary inhibition prediction task.
>
> **3. Emax:** An OpenADMET-provided dataset containing maximum-effect measurements. Models trained on this endpoint supplied additional embeddings capturing concentration-response-related molecular activity patterns.

### External Datasets

> **4. Therapeutics Data Commons (TDC) — Veith CYP inhibition:** Binary CYP inhibition classification data derived from the Veith screening datasets. These measurements were used to train endpoint-specific classification models and harvest representations learned from a substantially different experimental objective.
>
> **5. Novartis pIC50:** External quantitative CYP inhibition data containing pIC50 measurements for CYP2C9, CYP2D6, and CYP3A4. These measurements provided additional continuous inhibition labels for representation learning.

### External Dataset Selection and Scaffold Filtering

This emphasized chemical families with additional experimental evidence relevant to the test-set chemical space and also solved a time constraint for more time-consuming models by using a smaller training set.

> **6. TDC Veith scaffold filtering:** External Veith observations were restricted to generalized molecular scaffold families represented within the OpenADMET test set and having multiple observations within the external dataset.
>
> **7. Novartis pIC50 scaffold filtering:** The Novartis dataset was similarly restricted to generalized scaffold families represented in the OpenADMET test set and supported by multiple external observations.

These external scaffold-family filters were used to prioritize representation learning in chemical regions relevant to the challenge. Because the selection procedure used structural information from the test compounds, these experiments should be understood as test-aware, transductive representation learning rather than a strictly test-independent training protocol.

## 2. Model Training and Harvested Molecular Embeddings

### Neural Network Architectures

We trained multiple molecular learning architectures across the available datasets and applicable endpoints, extracting or inferring penultimate-layer molecular embeddings for downstream modeling.

The model families included:

> **Chemprop — Directed Message Passing Neural Network (D-MPNN):** Molecular graph models using directed bond-level message passing to learn representations of chemical structure and local molecular environments.
>
> **DeepChem AttentiveFP — Attentive Fingerprint Graph Neural Network:** Attention-based molecular graph models learning representations through iterative neighborhood aggregation and learned atomic attention mechanisms.
>
> **TorchMD-NET:** Three-dimensional molecular neural networks incorporating atomic coordinates and geometric information to learn conformationally informed molecular representations.
>
> **Uni-Mol:** Three-dimensional molecular representation models leveraging pretrained geometric molecular information and endpoint-specific adaptation.
>
> **SchNet — Continuous-Filter Convolutional Neural Network:** Geometry-aware molecular models using continuous-filter convolutions to learn representations from atomic identities and interatomic distances.

Models were trained independently across applicable combinations of molecular datasets, experimental endpoints, and CYP isoforms. This generated a diverse collection of learned molecular feature spaces, rather than a single representation shared across every prediction task.

![Molecular Representation Architecture](metacoder_architecture%20%282%29.png)

### Penultimate-Layer Embedding Extraction

For each trained model, we harvested molecular embeddings from the penultimate representation layer, prior to the final task-specific prediction output.

> Embeddings were extracted for molecules represented in the corresponding training datasets.
>
> Trained encoders were subsequently applied to the primary OpenADMET inhibition training set to infer molecular representations for additional compounds.
>
> Where applicable, the same trained encoders were used to generate representations for the blinded test set.
>
> Embeddings were aligned by molecular identity and original dataset row identifiers to maintain correspondence across independently trained architectures.
>
> For models requiring three-dimensional molecular conformers, we limited processing to **a maximum of three conformers per molecule**. Conformer-level representations were aggregated to produce a single molecular embedding for downstream modeling.

This procedure allowed representations learned from binary inhibition, single-concentration activity, Emax, and quantitative pIC50 endpoints to be reused in the primary CYP inhibition task.

### Additional Molecular Featurization

Alongside the neural-network embeddings, we incorporated conventional and pretrained molecular featurizers that did not depend on training a new model for each challenge endpoint.

These included:

> **RDKit molecular descriptors:** Physicochemical, topological, and structural descriptor variables.
>
> **CheMeleon molecular representations:** Pretrained molecular features providing an additional architecture-independent source of chemical information.
>
> **ECFP4 fingerprints:** Extended-connectivity fingerprints using radius 2, capturing local molecular substructures.
>
> **ECFP6 fingerprints:** Extended-connectivity fingerprints using radius 3, capturing larger molecular neighborhoods.

These features were evaluated independently and in combination with learned molecular embeddings to determine whether conventional structural information complemented the neural representations.

## 3. Ensemble Architectures and Multirepresentational Learning

Our modeling framework combined several prediction architectures to capture complementary information from independently trained models and datasets.

### A. Individual Model Predictions

The first ensemble component consisted of predictions generated directly by independently trained neural networks and conventional molecular models.

These included models trained specifically on the primary inhibition endpoint, as well as models trained on related CYP measurements where the prediction outputs could be meaningfully incorporated into the overall ensemble.

Different architectures capture distinct chemical relationships, providing a basis for ensemble diversity.

### B. Recombined Embedding Models with XGBoost

The principal multirepresentational approach involved concatenating harvested penultimate-layer embeddings from multiple independently trained encoders and fitting XGBoost regression models on the resulting feature spaces.

We constructed multiple variations of these combined representations, including:

> **Full multirepresentational supersets:** Broad combinations of available embeddings across architectures, datasets, and endpoints.
>
> **Graph-based representation combinations:** Features harvested from complementary two-dimensional molecular graph and message-passing architectures.
>
> **Three-dimensional representation combinations:** Geometry-aware embeddings from Uni-Mol, TorchMD-NET, and related conformational architectures.
>
> **Cross-dataset combinations:** Embeddings learned from inhibition, single-concentration, Emax, Veith, and Novartis datasets.
>
> **Cross-architecture combinations:** Selected mixtures of graph-based and three-dimensional representations.
>
> **Endpoint-specific subsets and ablations:** Alternative combinations designed to evaluate the contribution of particular training datasets, model families, or representation sources.

Each resulting feature matrix was used to train a downstream XGBoost model against the primary quantitative CYP inhibition endpoint.

This architecture separates molecular representation learning from final endpoint prediction. Rather than requiring a single neural network to learn every relevant chemical relationship from a relatively small inhibition dataset, XGBoost can exploit features previously learned across several larger or complementary molecular learning tasks.

### C. Pan-Isoform Representation Learning with VAE Compression

We also investigated a pan-isoform modeling architecture intended to learn shared relationships between molecular properties and CYP enzyme characteristics.

In this approach:

> Molecular embeddings harvested across CYP1A2, CYP2C9, CYP2D6, and CYP3A4 were assembled into broader cross-isoform molecular feature spaces.
>
> A **variational autoencoder (VAE)** was used to compress concatenated molecular embeddings into a lower-dimensional, unified latent representation.
>
> The compressed molecular representation was combined with tabular protein-level features describing the corresponding CYP isoform.
>
> Joint predictive models were used to investigate interactions between molecular properties and enzyme-specific characteristics.

The enzyme feature space followed the standard Metacoder pan-target representation framework and included:

> **ESM-2 protein embeddings:** Protein sequence representations derived from pretrained protein language models.
>
> **Structural and biophysical descriptors:** Tabular variables describing enzyme-level structural and physicochemical properties.
>
> **Multiple-sequence alignment (MSA) similarity features:** Principal-component-analysis (PCA) representations of protein sequence alignment similarities, providing a compact description of relationships between CYP isoforms.

This formulation allows a predictive model to consider both molecular representations and properties of the biological target, rather than treating every CYP isoform as an entirely independent prediction problem.

The pan-isoform architecture was investigated as a complementary approach to the individual CYP-specific models, with the longer-term objective of supporting broader pan-target molecular activity prediction.

### D. Out-of-Fold Evaluation and Model Selection

Scaffold-aware cross-validation was used to assess predictive generalization across structurally distinct molecular groups.

For the downstream XGBoost models, out-of-fold (OOF) predictions provided estimates of model performance and helped guide selection among alternative embedding combinations.

However, certain cross-dataset representation-learning experiments introduced potential information leakage. In particular, transferring embeddings from models trained on single-concentration measurements to the direct inhibition task could result in optimistic OOF estimates when related molecular observations or information were present across the datasets and validation partitions.

Consequently:

> OOF errors were used as approximate indicators of model quality rather than definitive estimates of external generalization.
>
> Models with potentially inflated OOF performance were not assumed to provide equivalent improvements on blinded data.
>
> Ensemble weighting also considered representation diversity, endpoint relationships, and the expected robustness of each component.
>
> Blinded challenge evaluation was treated as a more meaningful external assessment than potentially contaminated internal validation results.

## 4. Ensemble Blending and Distribution Adjustments

### A. Final Prediction Blending

The final submission combined predictions from several complementary sources:

> Individual neural-network models trained on CYP inhibition endpoints.
>
> XGBoost regressors trained on recombined molecular embedding sets.
>
> Models incorporating conventional molecular descriptors and pretrained representations.
>
> Selected pan-isoform and cross-endpoint modeling components.
>
> A minority contribution from externally generated structure-based predictions.

Ensemble weights were loosely informed by out-of-fold prediction errors, with adjustments reflecting model diversity, validation limitations, and the possibility of information leakage between related experimental datasets.

The objective was to balance models that captured different chemical relationships rather than simply assign the largest weights to models with the lowest observed internal validation error.

### B. Distribution-Based Prediction Adjustments

We incorporated distribution-adjustment concepts identified by **Briford**, extending these with additional prediction-flooring and low-activity correction rules.

These adjustments were motivated by the challenge's **ST-RAE evaluation metric**, which can create incentives for calibrated prediction behavior that differ from conventional regression optimization under RMSE or MAE.

Our postprocessing included:

> **Distribution adjustments:** Modifications informed by Briford's analysis of the predicted and expected inhibition distributions.
>
> **Soft-flooring:** Smooth adjustments intended to limit excessively low predicted pIC50 values without imposing a universal hard cutoff.
>
> **Low-activity rules:** Additional heuristic adjustments for predictions below approximately **4 pIC50**, intended to improve behavior in the low-inhibition regime.
>
> **Metric-aware calibration:** Evaluation of prediction adjustments in the context of ST-RAE rather than relying exclusively on conventional regression metrics.

These postprocessing procedures were applied to the ensemble predictions to improve calibration and align the final output with the competition's evaluation objective.

Because distribution-based and threshold-based corrections can introduce additional assumptions about the test-set response distribution, their benefits should be interpreted in the context of the challenge metric and the available validation evidence.

### C. Minority Blending of Scigantic Structure-Based Predictions

We additionally incorporated a minority-weight contribution from the structure-based, co-folding predictions provided by **Scigantic**.

Earlier in the competition, we had investigated implementing a Boltz-2-based protein–ligand co-folding pipeline as an additional source of CYP inhibition predictions. However, the computational demands of large-scale structure prediction, particularly without dedicated GPU resources, made comprehensive execution impractical within the competition timeline.

The availability of Scigantic's structure-based predictions provided an opportunity to introduce a complementary modeling perspective without independently reproducing the full co-folding workload.

These predictions were incorporated at a minority ensemble weight to diversify the final predictions while retaining the primary emphasis on Metacoder's molecular representation-learning and XGBoost-based modeling framework.

### D. Final Methodological Perspective

The central hypothesis of this submission is that **molecular representations learned independently across different neural architectures, experimental datasets, and biological endpoints can provide complementary information that improves downstream prediction when recombined into a common feature space.**

Our approach emphasizes reusable learned molecular features, cross-endpoint representation transfer, heterogeneous model architectures, and flexible downstream learning rather than reliance on a single end-to-end model.

The resulting framework combines graph-based molecular learning, three-dimensional geometric representations, conventional chemical descriptors, pan-isoform protein information, and structure-based prediction blending.

While limitations remain concerning cross-dataset leakage, validation calibration, computational constraints, and test-aware scaffold selection, the challenge provides a practical evaluation of multirepresentational learning for CYP inhibition prediction and a foundation for extending these methods to broader molecular–protein interaction modeling.
