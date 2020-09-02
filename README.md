# 501_project
Sentiment Analysis of Code-Mixed Social Media
Text (SemEval 2020 Task 9)
Task Description : https://competitions.codalab.org/competitions/20654

Team Members:
    Arnob Mallik (CCID: amallik)
    Arif Hasnat (CCID: hasnat)


Special Instructions:
	python 3.7+ required
	
	python3
	>>> import nltk
	>>> nltk.download('punkt')
	>>> nltk.download('averaged_perceptron_tagger')
	>>> nltk.download('wordnet')
	>>> nltk.download('sentiwordnet')
	>>> nltk.download('words')
	
	pip3 install numpy
	pip3 install pandas
	pip3 install scipy
	pip3 install sklearn
	pip3 install indic_transliteration -U
	pip3 install tensorflow
	pip3 install tensorflow_hub
	
	
Execution Instruction (running with semeval dataset):
    run the following command in command line:
        python3 code_mixed_semeval.py

	N.B. All data should be in data folder and python files should be in same directory as data folder
	
	
Execution Instruction (running with IIITH dataset):
    run the following command in command line:
        python3 code_mixed_iiith.py
		
	N.B. All data should be in data folder and python files should be in same directory as data folder


Execution Instruction (running naive_bayes on semeval dataset):		
	run the following command in command line:
        python3 naive_bayes.py TRAIN_FILE_PATH TEST_FILE_PATH

    example:
        python3 naive_bayes.py data/training_data.csv data/test_data.csv
