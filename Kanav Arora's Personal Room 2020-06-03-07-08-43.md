# # Assignment 2 Report

::Submitted By Kanav Arora (200439618)::

## Introduction: 
	The COVID19 pandemic has been devastating for hospitals as limited resources can be stretched. One are of work that is being investigated is the use of simulators that can determine the number of active cases likely to happen at a hospital.

## Objective 
	Using the above dataset, write a SparkML machine learning algorithm in order to predict one of the categories (for example fatality). 

## Deliverables
	1. Machine learning algorithm in Spark 
	2. Report on your findings
	
## Report
###  Library Imports

```
//*********************************************
/                                             \
*                                             *
* Library Imports                               *
*                                             *
/                                             \
//*********************************************


import org.apache.spark.sql.types.{ IntegerType, StructField, StructType, StringType }

import org.apache.spark.ml.feature.{ StringIndexer, VectorAssembler}

import org.apache.spark.ml.{Pipeline, PipelineModel}

import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}

import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
```

### Processing Raw Data

#### Loading Data

* I have loaded the data using spar.read function where the format of the data is csv and the file contains heads and `,` is the field separator acceding `HDFS` storage  of our clusters. 
```
val data = spark.read
  .format("csv")
  .option("header Indexed", value = true)
  .option("delimiter", ",")
  .option("mode", "DROPMALFORMED")
  .option("header", "true")
  .load("address")


// data leading
var covidDf = data.select(
  col("Assigned_ID").cast(IntegerType), 
  col("Age Group").cast(StringType),
  col("Ever Hospitalized").cast(StringType), 
  col("Ever in ICU").cast(StringType),
  col("Ever Intubated").cast(StringType),
  col("Currently Hospitalized").cast(StringType),
  col("Currently in ICU").cast(StringType),
  col("Currently Intubated").cast(StringType),
  col("Outcome").cast(StringType)
  )
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/Screenshot%202020-08-08%20at%2014.53.42.png)

* Filtering Data where `Outcome` is `FATAL`

```
val filterDF = covidDf
  .filter($"Outcome" === "FATAL")

filterDF.printSchema()

filterDF.show(10)
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/Screenshot%202020-08-08%20at%2014.54.52.png)


####  Indexing Features Columns
* We index  `Age Group, Ever in ICU, Ever Intubated, Currently Hospitalized ,  Currently in ICU,  Currently Intubated`  column to use for our machine learning algorithms
```
val IndexingFeatureCol = Array("Age Group", "Ever in ICU", "Ever Intubated", "Currently Hospitalized", "Currently in ICU", "Currently Intubated")

val IndexingFeatureCol = Array("Age Group", "Ever in ICU", "Ever Intubated", "Currently Hospitalized", "Currently in ICU", "Currently Intubated")
val indexersFeature = IndexingFeatureCol.map { colName =>
  new StringIndexer()
  .setInputCol(colName)
  .setOutputCol(colName + " Indexed")
  .setHandleInvalid("skip")
}

val featureIndexersPipeline = new Pipeline().setStages(indexersFeature)

val featureIndexedCovidDf = featureIndexersPipeline.fit(covidDf).transform(covidDf)

featureIndexedCovidDf.printSchema()
featureIndexedCovidDf.show(10)
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/Screenshot%202020-08-08%20at%2014.56.17.png)


#### Using VectorAssemble to combine cols to use as feature columns

::Scenario 1: All columns::  `Age Group Indexed", "Ever in ICU Indexed", "Ever Intubated Indexed", "Currently Hospitalized Indexed", "Currently in ICU Indexed", "Currently Intubated Indexed"` as feature columns

```
//feature cols for assembler
val featureCols = Array("Age Group Indexed", "Ever in ICU Indexed", "Ever Intubated Indexed", "Currently Hospitalized Indexed", "Currently in ICU Indexed", "Currently Intubated Indexed")
val featureCols = Array("Age Group Indexed", "Ever in ICU Indexed", "Ever Intubated Indexed", "Currently Hospitalized Indexed", "Currently in ICU Indexed", "Currently Intubated Indexed")
```

::Scenario 2: AgeGroup with Medical  History Only::
```
val featureCols = Array("Age Group Indexed", "Ever in ICU Indexed", "Ever Intubated Indexed")
```

::Scenario 3: AgeGroup with Current Condition Only::
```
val featureCols = Array("Age Group Indexed", "Currently Hospitalized Indexed", "Currently in ICU Indexed", "Currently Intubated Indexed")
```

* Vector Assemble to join columns
```
val assembler = new VectorAssembler()
  .setInputCols(featureCols)
  .setOutputCol("features")

val featureDf = assembler.transform(featureIndexedCovidDf)
featureDf.printSchema()

featureDf.show(10)

```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/B3B52EB9-4149-404A-B980-53A24509B3E8.png)


#### Indexing `outcome` as labels 
```
val indexer = new StringIndexer()
  .setInputCol("Outcome")
  .setOutputCol("label")
  .setHandleInvalid("skip")

val labelDf = indexer.fit(featureDf).transform(featureDf)
labelDf.printSchema()

labelDf.show(10)
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/CEE3419F-4BC5-42BC-9CC4-6786B5FCF3D6.png)

#### Splitting Data for Training and Transform
* seed = 5043
```
val seed = 5043
# Spliting into test and training data
val Array(trainingData, testData) = labelDf.randomSplit(Array(0.7, 0.3), seed)
````


### Machine Learning Algorithms

#### Random Decision Forest 
A decision tree is a supervised machine learning algorithm that can be used for both classification and regression problems. A decision tree is simply a series of sequential decisions made to reach a specific result. Here’s an illustration of a decision tree in action (using our above example):
![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/rfc_vs_dt11.png)

The decision tree algorithm is quite easy to understand and interpret. But often, a single tree is not sufficient for producing effective results. This is where the Random Forest algorithm comes into the picture.
![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/meme1.png)

```
val randomForestClassifier = new RandomForestClassifier()
  .setImpurity("gini")
  .setMaxDepth(3)
  .setNumTrees(20)
  .setFeatureSubsetStrategy("auto")
  .setSeed(seed)
```


###  Data Modelling

#### Building Random Forest Model   

* Fitting Data
```
// train Random Forest model with training data set
val randomForestModel = randomForestClassifier.fit(trainingData)

println(randomForestModel.toDebugString)
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/CCDE4E49-EFB5-4B0F-8C10-049AE6A0BFA2.png)

* Making Predictions
```
val predictionDf = randomForestModel.transform(testData)
predictionDf.show(10)
```

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/25F0DB79-C6BF-41B2-82DF-426ACF54E89C.png)

## Results

* Creating BinaryClassificationEvaluator
```
val evaluator = new BinaryClassificationEvaluator()
  .setLabelCol("label")
  .setMetricName("areaUnderROC")
```

### Accuracy
::Scenario 1: All columns:: 
Accuracy : 0.7648063240790789
![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/F147CF9F-8D39-45A9-9E3F-2684A42E1FE9.png)

::Scenario 2: AgeGroup with Medical  History Only:: 0.8053481810379538

![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/2F63DCCC-F147-4808-AC33-0F6A254B1C39.png)



::Scenario 3: AgeGroup with Current Condition Only::
Accuracy:  0.7479245618574984
![](Kanav%20Arora's%20Personal%20Room%202020-06-03-07-08-43/40A47E54-7516-4BF3-8D68-ED6C79C8A12C.png)

## Conclusion
Fatality prediction is better when we know from past history of the patient


# References & Citations
[1]  [Decision Tree vs. Random Forest – Which Algorithm Should you Use?](https://www.analyticsvidhya.com/blog/2020/05/decision-tree-vs-random-forest-algorithm/)



#Georgian/BigDataSem2/DataCollection&Curatioon/Assignment2/FinalReport
