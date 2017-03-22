# How to submit multiple .py files


## Question

Question and my answer here: 

* http://stackoverflow.com/questions/29036844/error-when-using-multiple-python-files-spark-submit/42921500#42921500


## Solution


**method-1**

Just putting `main.py` at the end of the cmd line.

    ../hadoop/spark-install/bin/spark-submit --master spark://spark-m:7077  --py-files /home/poiuytrez/naive.py,/home/poiuytrez/processing.py,/home/poiuytrez/settings.py main.py  



OR **method-2**

use `sc.addPyFile('py_file_name')` at `main.py`

    sc.addPyFile('/home/poiuytrez/naive.py')
    sc.addPyFile('/home/poiuytrez/processing.py')
    sc.addPyFile('/home/poiuytrez/settings.py')
