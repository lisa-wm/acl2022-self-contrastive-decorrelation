# Self-contrastive decorrelation with multiple subnetworks

Original proposal bei Klein & Nabi (2022, [arXiv:2203.07847v1](https://arxiv.org/pdf/2203.07847.pdf))

## Structure

* Main repository: forked from [repo for original paper](https://github.com/SAP-samples/acl2022-self-contrastive-decorrelation)
* Most important modifications
  * `train.py`: expand model to have multiple projection heads, add regularization term penalizing the standard deviation between the outputs of the former
  * `evaluation.py`: add uncertainty metrics to evaluation protocol
* Modifications imported from two submodules:
  * `SentEval`
    * Forked from Facebook's [SentEval toolkit](https://github.com/facebookresearch/SentEval)
    * Used for downstream evaluation
    * Contains additional folder `senteval_lisa` with implementations of different calibration error metrics
    * Separate because original evaluation protocol (`senteval`) is used in multiple places during training and modifying everything to match the additional implementations would be too much effort
  * `UncertaintyAwareSSL`
    * Forked from [here](https://github.com/amirvhd/Uncertainty_aware_SSL)
    * Contains custom losses and calibration error metrics

## Download & installation

1. Execute the following steps (your individual user rights might necessitate cloning with a private access token):
* `git clone git@github.com:lisa-wm/acl2022-self-contrastive-decorrelation.git`
* `cd acl2022-self-contrastive-decorrelation/`
* `git submodule init`
* `git submodule update`
* `sh data/download_wiki.sh && mv wiki1m_for_simcse.txt data/` 
* `cd SentEval/data/downstream/ && ./get_transfer_data.bash && cd ../../../`
* `git clone -b v4.10.0 https://github.com/huggingface/transformers.git scd_transformers`
* `cp transformers_v4.10/src/transformers/models/bert/modeling_bert.py scd_transformers/src/transformers/models/bert/`
* `cp transformers_v4.10/src/transformers/models/roberta/modeling_roberta.py scd_transformers/src/transformers/models/roberta/`
* `cd scd_transformers && pip install -e . && cd ..`

2. Create a virtual environment, e.g., via `conda env create -f env.yml` and `conda activate scdmh` 

## Usage

1. Run `bash run_train.sh` (possibly with modified config)
   * `alpha_unc`: coefficient on regularizer penalizing standard deviation between features output by different projector heads $\rightsquigarrow$ similarity
   * `lambda_unc`: cap on standard deviation $\rightsquigarrow$ diversity
2. Run `bash run_eval.sh` (possibly with modified config)
3. Find the results in the file specified as `--logfile` in `run_eval.sh` (e.g., `logfile.json`)