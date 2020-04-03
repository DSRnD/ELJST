# ELJST-Paper

## Instructions to run a new dataset.
- Make a copy of ELJST_Onetime and import the dataset and make the necessary preprocessings.
- Use the same ipynb to generate the edges.
- further, Test_ELJST_Onetime_only_edges.ipynb and Test_ELJST_Onetime_only_edges_attention_all.ipynb can be used to generate edges.
- attention_all can be generated only in attention_all.ipynb.
- all the edges will be in the resources folder.
- baselines can be run using the nohookup (nosampler_uni, nosampler_etm (mrf-lda), for etm there is a different repo).
- for lda, Baseline_LDA ipynb. Evaluations for LDA will be in the same file
- once the models are dumped, evaluation scipts should be run seperately.
- For MRF-LDA: Evaluation_ETM
- For ELJST: ELJST-Evaluation
- For ETM: Seperate Repo.
