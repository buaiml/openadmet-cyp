# OpenADMET Blind Challenge: Predicting CYP Inhibition

## Dataset preparation 

**Dataset 1**: Merge single concentraion dataset and 12-point dose-response curve (DRC) dataset across four major human CYP isoforms. As shown in the figure below, there is a correlation between the single-point data and the DRC data. Since the number of single points far exceeds that of DRC, we merged the two datasets, resulting in 8 tasks in total.
![correlation](./correlation-single-DRC.png) 
*Figure 1. Correlation between the single-point data and the DRC data* 

**Dataset 2**: Only the DRC dataset was used, yielding 4 tasks in total

**Dataset 3**: We named it as the "fillin" method. If the cyp*_log2fc_estimate of the single-point data is greater than the specified value, then cyp*_pIC50_direct_inhibiton = 2. The reason for this setting is that the proportion of measured DRC data for molecules with low activity is relatively small (Figure 1).

**Dataset 4 & 5**: Merge with the PubChem data. Assay IDs are AID 1851, AID 899,AID 410, AID 883,AID 891 and AID 884. They are all from National Center for Advancing Translational Sciences (NCATS). Dataset 4 uses the pubchem CYP data as a background, which means that pubchem endpoints are trained as separately tasks. Dataset 5 assumes that the PubChem data and the competition assay data are comparable, and therefore merges them together. It is worth noting that the PubChem dataset contains a large amount of inactive data. For these inactive samples, we assign a pIC50 value of 3. The figure below shows the correlation of pIC50 values for molecules shared between the competition and PubChem. As can be seen from the figure, the correlations for 2D6 and 1A2 are relatively high, while 3A4 and 2C9 have relatively few shared data points and show relatively low correlations.
![correlation](./pubchem-openadmet.png) 
*Figure 2. Correlation of pIC50 values for shared compounds (PubChem vs. OpenADMET)*

**Dataset 6**: Merge with the chembl data. For ChEMBL data, it comes from different literature and patents, and the experimental methods vary greatly. We expected a low possibility of improving prediction, so we only did a method similar to Dataset 5, that is, directly concatenating them with openADMET dataset together.




## Methods and results

Step 1: The same dataset, different AI algorithms, to find the best AI algorithm.
![results](./result1.png) 
*Table 1. Performance of different AI algorithms on Dataset 1.*

Step 2: Same algorithm for different datasets.
![results](./result2.png) 
*Table 2. Performance of unimol on differnt dataset*

## Discussion

In Step 1, Uni-Mol achieved the best overall performance. A plausible explanation is that Uni-Mol explicitly incorporates 3D conformational information and can therefore better capture the protein–ligand binding features. Meanwhile, the CYP2D6 model performed markedly worse than the others, and we further examined the underlying causes. As shown in Figure 3, the CYP2D6 assay correlates poorly with the other three assays, whereas the remaining three assays correlate well with one another. In addition, the binding pocket of CYP2D6 contains several anionic residues, and its binding characteristics therefore differ from those of the other isoforms. Furthermore, a different experimental method was used for CYP2D6 than for the other three isoforms.
![results](./CYP.cross.correlation.png) 
*Figure 3. Assay correlations*


**Key Findings from Step 2**

- **Finding 1: Merging single-concentration data improved predictive performance.** Combining the single-concentration data into the training set (Dataset 1 vs. Dataset 2) enhanced the overall predictive performance.

- **Finding 2: Imputing the activities of weakly active compounds (Dataset 3) improved performance on CYP1A2 and CYP2D6, particularly CYP2D6.** Filling in the activity values of weakly active compounds yielded notable gains for these two isoforms, with the most pronounced improvement observed for CYP2D6.

- **Finding 3: Direct vertical concatenation of the PubChem data into the training set improved predictive performance for CYP1A2 and CYP2D6.** Two plausible factors may account for this improvement:

  - First, as shown in Figure 2, the PubChem assay data for CYP1A2 and CYP2D6 correlate relatively well with the corresponding assays in the competition dataset, indicating good cross-source consistency.

  - Second, PubChem contains a large number of inactive compounds, whose activity values were likewise imputed. This expanded the pool of low-activity training samples, further supporting model performance on these isoforms.
 
- **Finding 4: Using the PubChem data as a background set (Dataset 4) slightly improved performance for CYP1A2, CYP2C9, and CYP3A4 relative to Dataset 1, although the gains were not significant; CYP2D6, in contrast, performed worse.** Incorporating the PubChem data as background samples yielded only modest gains for these three isoforms, whereas the CYP2D6 model showed degraded performance.

- **Finding 5: Incorporating the ChEMBL data did not contribute to predictive performance.**


After considerable deliberation over which CYP2D6 result to submit, we finally made our decision. As shown in Table 2, Dataset 3 (the fill-in method) achieved an MAE of 0.64, the best compared with both the leaderboard (broadcast baseline) result and our other results; however, its ST_RAE was less satisfactory. Dataset 5 (the method incorporating merged PubChem data) yielded a better RAE of 0.45, outperforming the fill-in method. For the midterm submission, we submitted the results of Dataset 5, which ranked second overall on the full dataset. For the final submission, however, we chose to prioritize MAE and therefore performed an ensemble of the above two results.


## Conclusion

We first identified the best-performing AI algorithm on a unified dataset, and then applied this algorithm to evaluate different data processing strategies. Imputing the activities of weakly active compounds and merging the PubChem data were found to improve predictive performance. Although these results are encouraging, the broader question of how to effectively leverage public datasets remains unresolved. In the present case, we were fortunate that PubChem provided a relatively large collection of data generated with assay methods comparable to those used in the competition. In most other cases, however, publicly available data resemble ChEMBL, where experimental protocols vary across institutions and are not directly comparable. How to make productive use of such heterogeneous data therefore remains an open question.
