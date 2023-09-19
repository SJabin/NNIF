# NNIFAdvTextDetector
This repository contains the codes for the paper "What Learned Representation and Influence Function Can Tell Us About Adversarial Examples".

## Environment set up
This repo supports `>=Python 3.7`. The required libraries can be installed using the `requirement.txt` file by runnning the below script:
```
pip install -r requirements.txt
```
## Fine-tuning LM

Run `fine_tune.py` as below:
```
python fine_tune.py --model-name="bert" --dataset-name="Mnli" --dataset-path="./multinli_1.0" --max-length=128
```
## Generate Adversarial Texts
Adversarial texts for this repo are generated using `generate_adv.py` (following [this](https://github.com/NaLiuAnna/MDRE) repo)
```
python generate_adv.py --dataset-name IMDB --dataset-path ./aclImdb --attack-class typo --max-length 512 --batch 0 --boxsize 25
```

## Running the detectors
Below adversarial text detection methods are implemented:

| Detector | |
|----------|---|
| NNIF | Image to text adaption https://github.com/giladcohen/NNIF_adv_defense|
| Mahalanobis | Image to text adaption https://github.com/pokaxpoka/deep_Mahalanobis_detector/tree/master|
| RSV| Adapted from https://github.com/JHL-HUST/RSV|
| SHAP| Adapted from https://github.com/huberl/adversarial_shap_detect_Repl4NLP/|
| MDRE| Using implementation from https://github.com/NaLiuAnna/MDRE|
| LID| Using implementation from https://github.com/NaLiuAnna/MDRE |
| FGWS| Using implementation from https://github.com/NaLiuAnna/MDRE |

After obtaining the fine-tuned LM and generating adversarial texts, run `detect.py` with the required arguments and observe the detector's performance. For example, use the below script to run the NNIF detector:
```
python detect.py --detect="nnif" --dataset-name="IMDB" --dataset-path="./aclImdb_v1/aclImdb" --attack-class="typo" --adv-path="./Pruthi_test_adv_IMDB.csv" --influence_on_decision --max-length=128 --start 0 --end 5000 --model-dir="./bert_imdb/" --max-indices=10
```
Or, run other detectors similar way:
```
python detect.py --detect="rsv" --dataset-name="Mnli" --dataset-path="./multinli_1.0" --attack-class="synonym" --adv-path="./Alzantot_test_adv_MNLI.csv" --transfer_out_file="./transfer_mnli_synonym.pkl" --max-length=128 --votenum=25
```
```
python detect.py --detect="shap" --dataset-name="Mnli" --dataset-path="./multinli_1.0" --attack-class="synonym"  --adv-path="./Alzantot_test_adv_MNLI.csv" --max-length=128
```



## Citation
Please use the below citation to cite our work:
```
TODO
```
