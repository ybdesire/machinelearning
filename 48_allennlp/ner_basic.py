from allennlp.predictors.predictor import Predictor

# Firstly, download model from https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz
predictor = Predictor.from_path("ner-model-2018.12.18.tar.gz")
result = predictor.predict(   sentence="Did Uriah honestly think he could beat The Legend of Zelda in under three hours?" )

print(result)


'''
{ 'tags': ['O', 'U-PER', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'L-MISC', 'O', 'O', 'O', 'O', 'O'], 
'words': ['Did', 'Uriah', 'honestly', 'think', 'he', 'could', 'beat', 'The', 'Legend', 'of', 'Zelda', 'in', 'under', 'three', 'hours', '?']}


U-PER: person
B: begining
I: inside the tag
'''







