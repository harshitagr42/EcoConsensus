# References and acknowledgements

Every external library, model, dataset and LLM used in this project. Copy this list into the README and the technical report. Entries marked **TO FILL** need information that is not available yet.

---

## Datasets

- **Pol-NLI** (training data). Burnham, M., Kahn, K., Wang, R. Y., and Peng, R. X. (2024). *Political DEBATE: Efficient Zero-shot and Few-shot Classifiers for Political Text.* arXiv:2409.02078. Dataset: Hugging Face `mlburnham/Pol_NLI`. The dataset page states no license; check the terms before publishing the data. This project uses a subset (3,517 training and 90 validation rows) produced by `data/raw/original/pol_nli_data/process_pol_nli.py`, with the `entailment` column read as 0 favour, 1 none, 2 against.
- **SemEval-2016 Task 6** (own test set and experiments only). Mohammad, S. M., Kiritchenko, S., Sobhani, P., Zhu, X., and Cherry, C. (2016). *SemEval-2016 Task 6: Detecting Stance in Tweets.* In Proceedings of the 10th International Workshop on Semantic Evaluation (SemEval-2016), pp. 31–41. Companion article: Mohammad, S. M., Sobhani, P., and Kiritchenko, S. (2017). *Stance and Sentiment in Tweets.* ACM Transactions on Internet Technology 17(3), Article 26. arXiv:1605.01655.
- **Hindi and Bengali versions** of both datasets are machine translations made once with an LLM. **TO FILL:** name and version of the translation LLM and the date.

## Models

- **MuRIL** (`google/muril-base-cased`). Khanuja, S. et al. (2021). *MuRIL: Multilingual Representations for Indian Languages.* arXiv:2103.10730. Google Research. Check the licence on the model card.
- **IndicBERT v2** (`ai4bharat/IndicBERTv2-MLM-only`). Doddapaneni, S., Aralikatte, R., Ramesh, G., Goyal, S., Khapra, M. M., Kunchukuttan, A., and Kumar, P. (2023). *Towards Leaving No Indic Language Behind: Building Monolingual Corpora, Benchmark and Models for Indic Languages.* In Proceedings of ACL 2023 (Long Papers), ACL Anthology 2023.acl-long.792. AI4Bharat. Check the licence on the model card.
- **LLM-A and LLM-B** (stance classification through APIs). **TO FILL:** provider, model name, model version or access date for each.

## Methods

- **LoRA.** Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR 2022. arXiv:2106.09685.
- **QLoRA** (attempted in the experiment; failed, reported as not run). Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. (2023). *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. arXiv:2305.14314.
- **AdamW.** Loshchilov, I. and Hutter, F. (2019). *Decoupled Weight Decay Regularization.* ICLR 2019. arXiv:1711.05101.
- **Dropout.** Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., and Salakhutdinov, R. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* Journal of Machine Learning Research 15, 1929–1958.
- **Stacked generalisation.** Wolpert, D. H. (1992). *Stacked Generalization.* Neural Networks 5(2), 241–259.
- **Matthews correlation coefficient.** Matthews, B. W. (1975). *Comparison of the predicted and observed secondary structure of T4 phage lysozyme.* Biochimica et Biophysica Acta 405(2), 442–451.
- **Cohen's kappa.** Cohen, J. (1960). *A coefficient of agreement for nominal scales.* Educational and Psychological Measurement 20(1), 37–46.

## Software

- **Python.** Python Software Foundation. https://www.python.org
- **PyTorch.** Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS 32.
- **Hugging Face Transformers.** Wolf, T. et al. (2020). *Transformers: State-of-the-Art Natural Language Processing.* In Proceedings of EMNLP 2020: System Demonstrations, pp. 38–45. Hugging Face.
- **PEFT.** Mangrulkar, S., Gugger, S., Debut, L., Belkada, Y., Paul, S., and Bossan, B. (2022). *PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods.* https://github.com/huggingface/peft
- **Accelerate.** Hugging Face. https://github.com/huggingface/accelerate
- **bitsandbytes** (training extras only). Dettmers, T. and contributors. https://github.com/bitsandbytes-foundation/bitsandbytes
- **scikit-learn.** Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* Journal of Machine Learning Research 12, 2825–2830.
- **pandas.** McKinney, W. (2010). *Data Structures for Statistical Computing in Python.* Proceedings of the 9th Python in Science Conference, 56–61.
- **NumPy.** Harris, C. R. et al. (2020). *Array programming with NumPy.* Nature 585, 357–362.
- **MLflow.** Zaharia, M. et al. (2018). *Accelerating the Machine Learning Lifecycle with MLflow.* IEEE Data Engineering Bulletin 41(4), 39–45. Databricks.
- **python-dotenv.** Kumar, S. and contributors. https://github.com/theskumar/python-dotenv
- **Hugging Face Hub** (model downloads). Hugging Face. https://huggingface.co

## Compute

- Google Colab (T4 GPU) for the experiment and training. **TO FILL:** add Kaggle if used for any run.

## Papers kept in `papers/` that are not yet cited here

**TO FILL:** `2212.05409v3.pdf`, `2306.13063v2.pdf`, `2602.21933v1.pdf`, `2024.case-1.33.pdf`, `2103.10730v2.pdf` (MuRIL, cited above), `1605.01655v1.pdf` (cited above), `2023.acl-long.792` (IndicBERT v2, cited above). Add a full citation for any of the remaining files that the report or README relies on.

## Course material

CSET 346 Natural Language Processing, Project Assignment: "Detecting Stance in Media on Global Warming for Indic Languages". Instructor: Dr. Supriya Chanda, Assistant Professor, SCSET.

## Writing and code assistance

**TO FILL if applicable:** the course asks students to mention LLMs used. State here whether an AI assistant helped with code or documents. Generated reports are produced by deterministic scripts and contain no AI-written text.
