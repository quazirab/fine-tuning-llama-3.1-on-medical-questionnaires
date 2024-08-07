# Fine-Tuning Llama 3.1 on medical questionnaires
Llama 3.1 is a great open-source Large Language Model. It is 3 different models. For this fine-tuning, the 8 billion parameter lightweight model was used due to the limitation of compute and GPU. Google Colab was used with T4 GPU and High RAM, and the unsloth accelerator was used for faster and more efficient fine-tuning. 

## Data Preparation
[notebook](notebooks/Llama_3_1_fine_tuning_with_LoRA.ipynb)

The first step was to pick the dataset and perform some data preparation. The dataset that was picked was Multiple Choice Questions, which would have to be prepared for the Question/Answer prompt. There were 4 columns with the choices and once column mentioning the correct choices column. There was also a column that explained the answer choice

For this training purpose, the questions that had single-choice instead of multiple-choice answers were selected. The answer was added to the new `answer` column. 

It was observed that there were repetitions of the answer choice number mentioned in the answer explanation column, so a data cleanup was required. Some patterns were identified for the repetition and using regex, the matched patterns were replaced. 
Since the number of rows was in the hundreds of thousands, for this exercise only 10,000 rows were chosen based on the explanation length

## Fine Tuning
[notebook](notebooks/data_prep_for_medical_question_training.ipynb)

Due to the ease of use with Google Colab, the unsloth accelerator was used. It is very begineer friendly and has great support with Hugging Face transformer library.


