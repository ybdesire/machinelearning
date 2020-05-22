# how to run spark code at aws emr


1. create emr spark cluster

2. notebook

3. new -> pyspark

4. we can use sc directly at pyspark notebook

5. run code below for example "word count"

```python
import pyspark

words = sc.parallelize ( ["scala", "java", "hadoop", "spark",  "akka", "spark vs hadoop", "pyspark", "pyspark and spark"]  )
counts = words.count()

print('word count={0}'.format(counts))
```