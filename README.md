# TNCSE
AAAI25 accepted paper
# Introduction

This repository is belong to the conference paper titled "TNCSE: Tensor's Norm Constraints for Unsupervised Contrastive Learning of Sentence Embeddings".

TNCSE is a BERT-like model for computing sentence embedding vectors, trained using unsupervised contrastive learning.

# How to Use

We recommend you train TNCSE with RAM >= 48GB and GPU memory >= 24GB.

## Installation

You also need to make sure your python >= 3.6 and install py repositories in requirements.txt :
```bash
pip install -r requirements.txt
```

After installation, make sure you download models' [checkpoint](https://drive.google.com/file/d/1sTrvx2dx0jtU77vH4uBoI7WVXDaZ0sXM/view?usp=drive_link) Google Drive and copy all the folders into the directory where the project resides. All the checkpoints you need are in these folders.

## Direct Evaluation

#### We report the results directly below the command.

### Eval TNCSE_BERT
```bash
python evaluation_CKPT.py --model_name_or_path_1 TNCSE_BERT_CKPT/BERT_1 --model_name_or_path_2 TNCSE_BERT_CKPT/BERT_2
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 75.52 | 83.91 | 77.57 | 84.97 | 80.42 | 81.72 | 72.97 | 79.58 |

### Eval TNCSE_RoBERTa
```bash
python evaluation_CKPT.py --model_name_or_path_1 TNCSE_RoBERTa_CKPT/RoBERTa1 --model_name_or_path_2 TNCSE_RoBERTa_CKPT/RoBERTa1
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 74.11 | 84.01 | 76.07 | 84.80 | 81.60 | 82.68 | 73.47 | 79.53 |

### Eval TNCSE_BERT_D
```bash
python evaluation_D.py --model_name_or_path TNCSE_D_BERT
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 75.42 | 84.64 | 77.62 | 84.92 | 80.50 | 81.79 | 73.52 | 79.77 |

### Eval TNCSE_RoBERTa_D
```bash
python evaluation_D.py --model_name_or_path TNCSE_D_RoBERTa
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 74.56 | 84.74 | 76.30 | 84.89 | 81.70 | 83.01 | 74.18 | 79.91 |

### Eval TNCSE_BERT_UC
```bash
python ensemble_UC.py --model_type BERT
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 75.80 | 85.27 | 78.67 | 85.99 | 82.01 | 83.16 | 73.01 | 80.56 |

### Eval TNCSE_RoBERTa_UC
```bash
python ensemble_UC.py --model_type RoBERTa
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 74.52 | 85.26 | 77.63 | 85.85 | 82.62 | 83.65 | 73.35 | 80.41 |

### Eval TNCSE_BERT_UC_D
```bash
python evaluation_D.py --model_name_or_path TNCSE_UC_D_BERT
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 75.94 | 85.31 | 78.50 | 85.69 | 81.86 | 83.03 | 73.89 | 80.60 |

### Eval TNCSE_RoBERTa_UC_D
```bash
python evaluation_D.py --model_name_or_path TNCSE_UC_D_RoBERTa
```

| STS12 | STS13 | STS14 | STS15 | STS16 | STSBenchmark | SICKRelatedness | Avg. |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 74.14 | 83.86 | 76.08 | 84.06 | 81.59 | 82.90 | 73.55 | 79.45 |

## Train TNCSE

### Data Preparation

We have prepared the unlabelled training set, located in **data/Wiki_for_TNCSE.txt**; the seven STS test sets are contained in **SentEval**.

### Pre-training Model

The checkpoints we need to use for training TNCSE-BERT and TNCSE-RoBERTa are saved in **TNCSE_BERT_encoder1**, **TNCSE_BERT_encoder2**, **TNCSE_RoBERTa_encoder1**, and **TNCSE_RoBERTa_encoder2**, respectively, which are RTT Data Augmentation and unsupervised SimCSE trained.

### Train TNCSE_BERT
```bash
python train_dual.py
```

### Train TNCSE_RoBERTa
```bash
python train_dual.py --output_path TNCSE_RoBERTa_OUTPUT --pretrain_model_path_1 TNCSE_RoBERTa_encoder1 --pretrain_model_path_2 TNCSE_RoBERTa_encoder2 --pretrain_tokenizer Roberta-base --batch_size_train 256 --lr 1e-06
```

#  Zero-shot downstream tasks evaluation

### The performance of the TNCSE and baseline models on 11 multilingual/cross-language semantic similarity tasks on the STS17 test set.

| **STS17** | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhinenedCSE** | **RankCSE** | **TNCSE** | **+D** | **+UC D** |
|:------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:---------------:|:------------------:|
| **ar-ar** | 48.98               | 52.72                | 51.00                | **54.99**       | 35.37              | 47.79                    | 48.79                | 53.63              | 53.00           | 54.63              |
| **en-ar** | -2.43               | 4.60                 | -4.86                | 0.13                 | 7.66               | -4.09                    | **11.73**       | -1.94              | -1.61           | 0.21               |
| **en-de** | 27.59               | 32.13                | 32.68                | 39.82                | 11.64              | 29.20                    | 28.90                | 37.95              | 36.12           | **39.68**     |
| **en-en** | 83.90               | 85.63                | 86.26                | 85.05                | 53.11              | 85.15                    | 85.88                | 85.41              | 85.36           | **86.38**     |
| **en-tr** | 10.58               | 10.67                | -6.25                | 6.83                 | 4.11               | -9.39                    | **15.58**       | 0.58               | 0.41            | 4.01               |
| **es-en** | 12.08               | 20.38                | 11.90                | 20.04                | 12.46              | 6.78                     | 13.22                | 20.16              | 18.96           | **22.46**     |
| **es-es** | 69.49               | 71.30                | 71.87                | **74.94**       | 56.06              | 73.27                    | 69.64                | 72.54              | 71.95           | 72.39              |
| **fr-en** | 36.18               | 28.08                | 26.71                | 27.47                | 12.28              | 26.50                    | **43.60**       | 40.03              | 39.49           | 39.29              |
| **it-en** | 20.08               | 15.72                | 11.75                | 12.63                | 6.92               | 13.36                    | **23.39**       | 16.73              | 14.61           | 18.51              |
| **ko-ko** | **52.62**      | 49.67                | 52.00                | 51.06                | 39.64              | 51.00                    | 52.58                | 52.05              | 50.86           | 51.96              |
| **nl-en** | 17.40               | 21.10                | 14.65                | 22.50                | 9.77               | 16.34                    | 20.20                | 22.39              | 20.38           | **25.52**     |
| **Avg.**  | 34.22               | 35.64                | 31.61                | 35.95                | 22.64              | 30.54                    | 37.59                | 36.32              | 35.41           | **37.73**     |

### The performance of the TNCSE and baseline models on 18 multilingual/cross-language semantic similarity tasks on the STS22 test set.

| **STS22** | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **TNCSE** | **+D**    | **+UC D** |
|:--------------------------------------------------------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:---------------:|:------------------:|
| **ar**    | **38.33**  | 32.48            | 34.94            | 21.08            | 33.58          | 36.08          | 35.33            | 31.26          | 32.88          | 34.03          |
| **de**    | 24.70           | 28.50            | 24.47            | 18.02            | 2.58           | 24.99          | 24.70            | 28.04          | 27.75          | **29.18** |
| **de-en** | 13.13           | 29.80            | 33.63            | **37.03**   | 20.78          | 30.33                | 35.51            | 33.11          | 31.83          | 34.89          |
| **de-fr** | 35.92           | 32.68            | 38.29            | 2.44             | 25.42          | 31.45                | **39.27**   | 35.12    | 39.16          | 33.72          |
| **de-pl** | 18.82     | 12.78            | 11.30            | -26.67           | 7.08           | 9.58                 | 5.67             | **28.52** | 26.64          | 23.06          |
| **en**    | 59.11           | 60.66            | 61.15      | 54.96            | 54.23          | 60.16                | 62.46            | **63.23** | **63.79** | **63.28** |
| **es**    | 49.23           | 52.14            | 55.03            | 49.06            | 39.98          | 55.16          | 54.96            | **58.14** | **58.14** | **58.43** |
| **es-en** | 30.44           | 37.84            | 36.83            | 38.53            | 21.28          | 34.14                | 38.50            | **39.70** | 37.45          | **39.57** |
| **es-it** | 31.48           | 42.50            | 40.91            | 44.44            | 22.54    | 31.27                | 42.16            | 43.34          | 44.28          | **44.58** |
| **fr**    | 61.55     | 61.31            | 60.06            | 52.95            | 31.47          | 52.96                | 65.35            | **66.84** | **66.36** | **66.51** |
| **fr-pl** | 39.44     | **50.71**   | -5.63            | 16.90            | 16.90          | 16.90                | 39.44            | 39.44          | 39.44          | 39.44          |
| **it**    | 54.67           | 59.89      | 57.61            | 52.94            | 27.64          | 53.46                | 60.60            | **63.98** | **63.31** | **62.98** |
| **pl**    | 22.79           | 26.72            | 23.77      | 8.23             | 6.78           | 23.42                | 26.09            | 25.72          | 25.79          | **27.13** |
| **pl-en** | 15.44           | **36.41**   | 30.43      | 29.48            | 28.67          | 22.82                | 33.62            | 35.19          | 30.77          | 31.79          |
| **ru**    | 15.71           | 17.87            | 24.03      | 6.77             | 14.03          | **24.59**       | 18.89            | 20.37          | 20.09          | 20.22          |
| **tr**    | 28.09           | 31.56            | 29.18            | 24.27            | 16.92          | 28.33                | 28.61            | 34.01    | **34.42** | **32.61** |
| **zh**    | 46.42           | 37.76            | **48.78**   | 47.06            | 40.12    | 40.45                | 46.38            | 43.93          | 41.83          | 41.65          |
| **zh-en** | 4.82            | 9.87             | 13.14            | **27.61**   | 15.06          | 11.94                | 8.60             | 18.22          | 17.94          | 23.58          |
| **Avg.**  | 32.78           | 36.75            | 34.33            | 26.48            | 23.61          | 32.67                | 37.01            | **39.34** | **38.95** | **39.26** |

### We report TNCSE and baselines performance on 30 classification tasks.

| **Tasks**                                                   | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhinenedCSE** | **RankCSE** | **TNCSE** | **+D** | **+UC D** |
|:--------------------------------------------------------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:---------------:|:------------------:|
| **Banking77Classification**                                 | 74.43               | 73.87                | 76.09                | **78.17**       | 65.88              | 75.59                    | 75.69                | 76.10              | 75.98           | 77.22              |
| **BulgarianStoreReviewSentimentClassfication**              | 29.29               | 30.44                | 31.92                | 32.58                | **34.51**     | 31.54                    | 31.54                | 30.00              | 29.56           | 30.22              |
| **CSFDCZMovieReviewSentimentClassification**                | 20.32               | 20.28                | 20.91                | 20.92                | 20.11              | 21.21                    | 20.71                | 21.03              | 21.07           | **21.30**     |
| **CUADAffiliateLicenseLicensorLegalBenchClassification**    | 78.41               | 75.00                | **85.23**       | 76.14                | 70.45              | 81.82                    | 72.73                | 70.45              | 68.18           | 72.73              |
| **CUADChangeOfControlLegalBenchClassification**             | **71.15**      | 63.70                | 69.71                | 67.79                | 71.15              | 67.07                    | 66.11                | 69.95              | 68.99           | 68.03              |
| **CUADInsuranceLegalBenchClassification**                   | **94.17**      | 88.64                | 91.36                | 91.84                | 89.51              | 86.02                    | 90.78                | 89.42              | 90.00           | 92.62              |
| **CUADMostFavoredNationLegalBenchClassification**           | 78.13               | **84.38**       | 75.00                | 75.00                | 64.06              | 76.56                    | 71.88                | 79.69              | 78.13           | 76.56              |
| **CUADThirdPartyBeneficiaryLegalBenchClassification**       | 77.94               | 83.82                | 77.94                | **85.29**       | 73.53              | 82.35                    | 80.88                | 77.94              | 80.88           | 79.41              |
| **Diversity2LegalBenchClassification**                      | 74.00               | 73.67                | 74.33                | 74.67                | **76.00**     | 74.67                    | 74.67                | 74.33              | 74.33           | 74.67              |
| **Diversity5LegalBenchClassification**                      | 60.00               | 61.00                | 60.67                | 61.67                | **88.33**     | 59.67                    | 63.00                | 62.00              | 61.33           | 61.00              |
| **FunctionOfDecisionSectionLegalBenchClassification**       | 16.35               | 15.26                | 15.80                | 15.53                | **23.16**     | 16.35                    | 14.71                | 17.44              | 17.44           | 14.44              |
| **GermanPoliticiansTwitterSentimentClassification**         | 38.21               | 36.67                | 37.98                | 39.41                | **40.14**     | 38.68                    | 38.74                | 38.01              | 37.45           | 37.65              |
| **IndonesianIdClickbaitClassification**                     | 54.26               | 53.89                | 54.09                | 54.56                | **57.57**     | 54.15                    | 53.44                | 54.68              | 55.02           | 55.36              |
| **Itacola**                                                 | 48.56               | 49.02                | 47.28                | **50.62**       | 50.07              | 47.82                    | 48.13                | 48.87              | 48.94           | 49.64              |
| **JCrewBlockerLegalBenchClassification**                    | 77.78               | 74.07                | 72.22                | 57.41                | **87.04**              | 72.22                    | 75.93                | 66.67              | 72.22           | 68.52              |
| **LearnedHandsDivorceLegalBenchClassification**             | 76.00               | 79.33                | 75.33                | 69.33                | 64.67              | 80.67                    | 83.33                | **84.00**     | **84.00**  | 83.33              |
| **LearnedHandsDomesticViolenceLegalBenchClassification**    | 78.16               | 77.59                | **78.74**       | 70.69                | 72.41              | 75.86                    | 75.29                | 78.16              | 77.01           | 76.44              |
| **LearnedHandsFamilyLegalBenchClassification**              | 70.75               | 77.10                | 68.65                | 71.48                | 64.99              | 68.85                    | 71.53                | **79.20**     | **80.66**  | **78.52**     |
| **LearnedHandsHealthLegalBenchClassification**              | 57.52               | **66.81**       | 60.18                | 62.83                | 59.29              | 62.83                    | 66.81                | 64.16              | 63.72           | 65.04              |
| **LegalReasoningCausalityLegalBenchClassification**         | 65.45               | 61.82                | 58.18                | 60.00                | **74.55**     | 63.64                    | 67.27                | 63.64              | 69.09           | 69.09              |
| **MacedonianTweetSentimentClassification**                  | 35.72               | 36.09                | 37.44                | 36.74                | **37.94**     | 37.16                    | 36.50                | 36.89              | 36.72           | 37.38              |
| **OPP115UserAccessEditAndDeletionLegalBenchClassification** | 62.99               | 64.72                | 66.23                | **72.51**                | 60.61              | 67.53                    | 67.53                | 64.72              | 64.72           | 67.97              |
| **OralArgumentQuestionPurposeLegalBenchClassification**     | 22.44               | 15.71                | 21.47                | 19.87                | 24.04              | 23.08                    | 21.79                | **25.64**     | **25.00**  | **25.00**     |
| **PersianFoodSentimentClassification**                      | 55.98               | 55.76                | 56.77                | 58.01                | 53.55              | 57.43                    | 55.95                | 57.25              | 57.18           | **57.60**     |
| **RestaurantReviewSentimentClassification**                 | 50.59               | 51.39                | 50.29                | 50.72                | **53.03**              | 51.88                    | 50.10                | 52.04     | 51.60           | 51.77              |
| **SCDDCertificationLegalBenchClassification**               | 67.46               | 70.63                | 59.26                | **73.81**       | 66.14              | 57.94                    | 58.20                | 63.49              | 64.29           | 65.34              |
| **SCDDTrainingLegalBenchClassification**                    | 58.31               | 59.10                | 50.66                | **63.32**       | 46.44              | 48.28                    | 60.42                | 53.03              | 53.56           | 54.09              |
| **ToxicChatClassification**                                 | 69.47               | 68.50                | 66.31                | 71.01                | 69.78              | 64.60                    | 68.21                | 71.32              | 68.57           | **71.41**     |
| **TweetEmotionClassification**                              | 26.48               | 27.20                | 28.53                | 28.30                | **29.37**     | 28.59                    | 26.07                | 27.10              | 27.12           | 27.54              |
| **TweetSentimentExtractionClassification**                  | 54.27               | 52.36                | 54.54                | 54.23                | 53.25              | 53.91                    | **56.53**       | 54.30              | 54.95           | 55.31              |
| **Avg.**                                                    | 58.15               | 58.26                | 57.44                | 58.15                | 58.05              | 57.60                    | 58.15                | **58.38**     | **58.59**  | **58.84**     |

### We report TNCSE and baselines performance on 30 retrieval tasks.
| **Tasks**                           | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **TNCSE** | **TNCSE D** | **TNCSE UC D** |
|:-----------------------------------:|:----------:|:-----------:|:-----------:|:-----------:|:---------:|:---------------:|:-----------:|:---------:|:-----------:|:--------------:|
| **AILAStatutes**                    | 8.17       | **9.78**        | 7.03        | 8.83        | 7.43      | 6.25            | 6.58        | 9.36      | 8.84        | 8.55           |
| **AlloprofRetrieval**               | 3.68       | 6.27        | 2.71        | 2.74        | 2.67      | 1.56            | 3.57        | 6.13      | **6.46**        | 5.26           |
| **ArguAna-PL**                      | 5.47       | 6.65        | 6.76        | 8.05        | 6.58      | 6.25            | 4.76        | 7.39      | 7.30        | **8.15**           |
| **CQADupstackEnglishRetrieval**     | 16.78      | 16.16       | 16.61       | **22.06**       | 13.27     | 16.68           | 15.49       | 15.25     | 16.31       | 19.64          |
| **CQADupstackGamingRetrieval**      | 23.83      | 23.51       | 25.41       | **31.13**       | 20.73     | 25.18           | 22.34       | 24.92     | 25.25       | 29.20          |
| **CQADupstackMathematicaRetrieval** | 4.39       | 5.33        | 5.26        | 6.89        | 4.41      | 4.88            | 5.14        | 6.08      | 5.98        | **7.72**           |
| **CQADupstackTexRetrieval**         | 4.42       | 6.40        | 6.21        | 8.44        | 5.77      | 6.38            | 5.48        | 6.57      | 6.72        | **8.55**           |
| **CQADupstackWebmastersRetrieval**  | 13.32      | 13.41       | 14.16       | 16.00       | 13.49     | 14.89           | 12.20       | 14.31     | 15.11       | **17.33**          |
| **EstQA**                           | 21.14      | 31.41       | 25.73       | 28.58       | 17.63     | 22.02           | 19.55       | 31.17     | **32.43**       | **31.70**          |
| **FQuADRetrieval**                  | 27.77      | 25.71       | 24.55       | **31.10**       | 19.41     | 17.30           | 28.04       | 27.91     | 28.58       | 30.05          |
| **GeorgianFAQRetrieval**            | 2.88       | 2.59        | 3.45        | 2.85        | 3.03      | 3.16            | 2.35        | 3.08      | 3.40        | **3.66**           |
| **GerDaLIRSmall**                   | 1.65       | 1.93        | 1.60        | 1.72        | 1.60      | 1.38            | 1.32        | **1.94**      | **1.95**        | **2.12**           |
| **GermanGovServiceRetrieval**       | 19.73      | 28.04       | 19.45       | 36.45       | 12.49     | 15.37           | 18.23       | 28.**50**     | 27.15       | **29.20**          |
| **GermanQuAD-Retrieval**            | 26.60      | 37.19       | 22.51       | 22.42       | 24.00     | 11.63           | 29.97       | **38.45**     | **38.21**       | 36.68          |
| **JaQuADRetrieval**                 | 3.00       | 3.16        | 2.72        | 4.01        | 3.21      | 3.27            | 3.89        | **4.99**      | **4.53**        | **4.40**           |
| **LegalBenchCorporateLobbying**     | 77.23      | **78.34**       | 77.50       | 78.02       | 74.84     | 74.41           | 69.61       | 77.18     | 77.10       | 78.18          |
| **LegalSummarization**              | 45.57      | 45.80       | 44.45       | 46.20       | 43.43     | 43.29           | 44.25       | **48.62**     | **48.30**       | **50.96**          |
| **LEMBNarrativeQARetrieval**        | 8.63       | 8.96        | 8.36        | 8.98        | 7.29      | 8.20            | 9.93        | **11.90**     | **11.13**       | **11.35**          |
| **LEMBQMSumRetrieval**              | 9.64       | 11.54       | 10.10       | 10.82       | 9.47      | 9.09            | 10.84       | **12.82**     | **12.47**       | **13.12**          |
| **LEMBSummScreenFDRetrieval**       | 39.88      | 45.34       | 41.75       | 38.93       | 28.58     | 41.46           | 40.40       | **52.69**     | **51.54**       | **52.36**          |
| **LEMBWikimQARetrieval**            | 27.48      | 26.72       | 25.67       | 31.47       | 27.86     | 25.77           | 25.63       | 29.38     | 28.50       | **31.61**          |
| **MedicalQARetrieval**              | 20.25      | 18.06       | 17.35       | 27.73       | 17.66     | **22.09**           | 19.20       | 21.47     | 21.81       | 21.30          |
| **NFCorpus**                        | 3.66       | 3.55        | 3.43        | 4.44        | 2.51      | 3.81            | 2.10        | **3.97**      | 3.70        | 3.67           |
| **NQ**                              | 9.79       | 8.53        | 9.50        | **15.84**       | 9.04      | 9.41            | 8.51        | 8.92      | 10.58       | 10.93          |
| **QuoraRetrieval**                  | 75.58      | 74.23       | 75.66       | 77.57       | 70.06     | 74.54           | 76.41       | 77.08     | 76.76       | **78.21**          |
| **SCIDOCS**                         | 4.22       | 4.26        | 4.45        | **5.06**        | 3.97      | 4.17            | 3.01        | 4.37      | 4.55        | 4.68           |
| **SciFact**                         | 32.07      | 34.37       | 33.05       | **34.83**       | 27.34     | 32.22           | 23.13       | 30.65     | 29.47       | 33.51          |
| **SlovakSumRetrieval**              | 5.51       | 24.52       | 10.18       | 8.71        | 7.00      | 5.68            | 12.13       | **29.96**     | **30.40**       | **27.86**          |
| **SyntecRetrieval**                 | 26.56      | 31.72       | 26.42       | 25.72       | 21.15     | 18.91           | 25.49       | 27.52     | 29.19       | **34.37**          |
| **TRECCOVID**                       | 0.64       | 0.56        | **0.65**        | 0.63        | 0.56      | 0.61            | 0.48        | 0.53      | 0.51        | 0.54           |
| **Avg.**                            | 18.98      | 21.13       | 19.09       | 21.54       | 16.88     | 17.66           | 18.33       | **22.10**     | **22.14**       | **23.16**          |


### We report TNCSE and baselines performance on nine reranking tasks.
| **Model**                     | **SimCSE** | **ESimCSE** | **DiffCSE** | **InfoCSE** | **SNCSE** | **WhitenedCSE** | **RankCSE** | **TNCSE** | **TNCSE D** | **TNCSE UC D** |
|:--------------------------------------:|:-------------------:|:--------------------:|:--------------------:|:--------------------:|:------------------:|:------------------------:|:--------------------:|:------------------:|:--------------------:|:-----------------------:|
| **AlloprofReranking**         | 30.46               | 32.74                | 28.33                | 27.97                | 25.31              | 27.34                    | 29.25                | **33.63**     | 31.76                | **33.94**          |
| **AskUbuntuDupQuestions**     | 51.88               | 52.28                | 52.08                | 52.83                | 45.53              | 51.60                    | **53.76**       | 52.44              | 53.65                | 52.56                   |
| **CMedQAv1**                  | 13.07               | 13.63                | 14.05                | **17.23**       | 11.01              | 14.61                    | 14.04                | 14.42              | 15.61                | 14.25                   |
| **CMedQAv2-reranking**        | 13.97               | 14.78                | 15.26                | **17.21**       | 11.69              | 15.06                    | 14.47                | 14.42              | 16.01                | 14.68                   |
| **MMarcoReranking**           | 2.48                | 3.77                 | 3.64                 | **4.96**        | 2.70               | 4.02                     | 3.34                 | 3.54               | 4.32                 | 3.51                    |
| **SciDocsRR**                 | 67.87               | 70.48                | 70.37                | 71.29                | 58.90              | 67.63                    | 69.89                | 69.87              | **71.47**       | 69.74                   |
| **StackOverflowDupQuestions** | 39.57               | 40.64                | 42.77                | **44.21**       | 31.06              | 42.64                    | 41.18                | 42.05              | 43.25                | 41.94                   |
| **SyntecReranking**           | 56.53               | **58.92**       | 51.37                | 52.87                | 49.25              | 53.98                    | 52.43                | 56.55              | 57.93                | 57.40                   |
| **T2Reranking**               | 55.20               | 55.87                | 56.27                | **56.71**       | 52.10              | 56.16                    | 55.59                | 56.44              | 56.41                | 56.61                   |
| **Avg.**                      | 36.78               | 38.12                | 37.13                | 38.36                | 31.95              | 37.01                    | 37.11                | 38.15              | **38.93**       | 38.29                   |
