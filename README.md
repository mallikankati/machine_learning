## Machine Learning Algorithms Implementation

My attempt to implement ML algorithms in python, panda, numpy and sklearn

I am not going to implement all algorithms but will try to cover simplest algorithms first before dig into complex one

Highly recommend Udemy Machine Learning course by [SuperScienceData Team](https://www.udemy.com/machinelearning/). They have amazing content and I learned Machine Learning from this course.

**Note:**  All algorithms require a standard data preprocessing step which covered in details

**1: [Data Preprocessing](1_Data_Preprocessing)**

Any ML algorithm require a data preprocessing (DP). In this section we cover why do we need DP and for implementation details look under [Data Preprocessing](1_Data_Preprocessing)

Why do we need Data Preprocessing

  - data file may comes with missing data, in that case come up with uniform replacement (or dropping row ) for missing data in a given column.
  - If the data contains `text` and `numbers`, need to convert `text` related columns to numbers before feed into ML
  - If the data is not uniform on given column, need to scale that column by applying `Normalization` or `MinMax`
  - If the column is not used for training and prediction drop it before applying ML algorithm

**2. [Regression](2_Regression)**
   
   [Sunil Ray Blog](https://www.analyticsvidhya.com/blog/2015/08/comprehensive-guide-regression/) explains different kinds of regression. Please read for refreshing basics on regressions. I implemented following regression algorithms
   
   - [Simple Linear Regression](2_Regression/1_Linear_Regression)
   - [Polynomial Regression](2_Regression/2_Polynomial_Regression)
   - [Support Vector Regression](2_Regression/3_Support_Vector_Regression)
   - Decision Tree Regression
   - Random Forest Regression
   
**3. [Classification](3_Classification)**  

   - [Logistic Regression](3_Classification/Logistic_Regression)

**4. Clustering**

**5. Reinforcement Learning**

**6. Natural Language processing**

**7. Dimensionality Reduction**

**Note:** Deep learning we will cover in as a different projects 
