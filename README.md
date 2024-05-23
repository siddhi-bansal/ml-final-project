# Refining Ratings: Using Machine Learning to Predicting Restaurant Star Ratings
**Group: Siddhi Bansal, Isha Kalanee, Ishita Kumari, Akshat Shah**

Our goal was to train a machine-learning model that predicts a restaurant’s star rating based on specific restaurant attributes (total_open_hours, RestaurantsTakeOut, RestaurantsDelivery, Alcohol). This model can be used by new restaurant businesses to predict their star rating (as a measure of success) using the specific features required by the model

Our chosen dataset is accessible here: https://yelp.com/dataset/download. Specifically, we used the  `yelp_academic_dataset_business.json` file.

### I. Data Preparation

Our original dataset contains 150,346 entries of businesses recorded on Yelp, each record containing attributes such as `name`, `address`, `latitude`, `longitude`, `review_count`, and `stars`. Since we want our model to focus on predicting the success of restaurants, we only keep the businesses that are labeled as `Restaurants`. Additionally, we drop the columns that are irrelevant and those that will bias/skew our models, such as `name` and `business_id`. We are choosing to deal with NaNs by dropping all records with NaN values in the original dataset and further downsizing our dataset by dropping all businesses that are no longer open.

### II. Data Exploration

We searched for correlations between various combinations of the remaining attributes. Eventually, we visualized the locations of the restaurants on a Cartesian plane by `latitude` (horizontal axis) and `longitude` (vertical axis), features of the original dataset. An important observation we made is that the data included information about the business’s daily hours. Another is determining how many missing values each feature had. This is critical to choose which columns to remove (to also reduce dimensionality) and maintain a dataset good enough for our training purposes. There were quite a few missing values in many of the columns created (from the `attributes` column) during data preparation since the data was initially unstructured JSON.

### III. Feature Engineering

After exploring the data, we sought to create a final dataframe by modifying or aggregating certain features and leveraging the stacking technique to create an ensemble.

We started by creating our own function to parse each day’s hours. While the hours were in 24-hour format, there were some inconsistencies within the data. We made a few basic assumptions such as the following which are accounted for in the code:

- Missing value indicates the restaurant is closed that day
- If the end time is past 23:59, it’s on the next day
- If the start and end times are the same (e.g. 0:0-0:0), then the restaurant is open for all 24 hours of that day

The `parse_hours()` function implements all of the above; we sum each day’s hours into one value, stored in the total_open_hours column as the total weekly hours. We also removed each day’s hours columns, reducing the dimensionality of the data.

Secondly, we simplify the alcohol column, which indicates what type of alcohol the restaurant serves, if any. We made some more basic assumptions in doing this, such as a missing value indicating no alcohol. To simplify the data, we converted all values to true and false, rather than the more specific categories such as `beer_and_wine` and `full_bar`.

The stacking that we added at this stage includes:
'stack_1': A KNN Regression model on our existing dataset: estimates the business’s average rating by finding the mean rating of its k-nearest neighbors. The successful location-based clustering from data exploration led us to believe that there tends to be a slight correlation between the relative location of a restaurant and its ratings. We decided that training our model with this additional column could help improve accuracy.
'stack_2': A Decision Tree Regressor: predicts the number of stars based on stack 1 and the current four attributes: `total_open_hours`, `RestaurantsTakeOut`, `RestaurantsDelivery`, and `Alcohol`, with `stars` as the label class.

Lastly, we imputed any missing values in our 3 boolean features as False.

- Weekly hours open – sum of hours for each day
- Broader categories for alcohol – multiple classes to just true/false (boolean)
- Stacking
  - For each restaurant, use KNN to find k nearest restaurants and compute their average rating → add as feature into dataframe
  - For each restaurant, use a Decision Tree Regressor to predict the number of stars based on four features (`total_open_hours`, `RestaurantsTakeOut`, `RestaurantsDelivery`, `Alcohol`) 
    and the label as `stars` (since this is a supervised algorithm)

Our final dataframe contains the following features:
- `RestaurantsTakeOut` - True if the Restaurant offers takeout, False otherwise
- `RestaurantsDelivery` - True if the Restaurant offers delivery, False otherwise
- `Alcohol` - True if the Restaurant serves alcohol, False otherwise
- `total_open_hours` - The total number of hours the restaurant is open every week
- `stack_1` - Star predictions using a KNN Regression Model
- `stack_2` - Star predictions using a Decision Tree Regressor

We use the `stars` column as the label.

### IV. Modeling

When training potential models, we tested each model with all combinations of stacks (features with stack 1, features with stack 2, features with stacks 1 and 2, and features with neither stacks 1 or 2) to determine which yields the highest accuracy.

The models we tested in search of the best accuracy include:
- Linear Regression: Uses the sklearn LinearRegression model and performs 10-fold cross-validation (CV). Yields a maximum accuracy of 10.09% when trained on the dataset with only stack 2.
- Decision Tree Regressor: Uses the sklearn DecisionTreeRegressor and performs a 10-fold CV. Returns a maximum accuracy of 9.91% when trained on the dataset with only stack 2.
- Neural Nets Regressor: Uses the MLPRegressor with a pipeline and GridSearchCV to find the optimal combination of hyperparameters. We then train the model with a 5-fold CV, which results in a maximum score of -8.04 with the stack 2 dataset.
- KNN Classifier:  Used the KNeighborsClassifier from sklearn and performs a 5-fold cross validation. Results in a maximum accuracy of 27.11% when trained on the dataset with only stack 2.
- Decision Tree Classifier: Uses the sklearn DecisionTreeClassifier and performs 5-fold CV, yielding a max accuracy of 27.84% when using the dataset with only stack 2.
- Neural Nets Classifier: Used a pipeline consisting of a standard scalar and the MLPClassifier from sklearn with a hidden layer size of 30. GridSearchCV is used to find the optimal hyperparameters, then we do a 5-fold cross validation and a maximum accuracy of 27.12% when trained on the dataset with only stack 1.
- K-Means Classification: Used the KMeans clustering model to classify all entries into 9 clusters (the range of ‘stars’ is 1.0 - 4.5 at 0.5 intervals). Yielded the best (minimum) MSE of 0.7353 when using the stack 2 dataset.

#### K-Means Silhouette and MSE Values by Dataset

We concluded that K-Means Clustering is the best model to use for our data, since it demonstrates an MSE of 0.7353 and silhouette score of 0.8916. This model performed significantly better than the rest that we tested.

| features                | Avg Silhouette Score | Avg MSE            |
| ----------------------- | ---------------------| ------------------ |
| features_all_stacks     | 0.8890943569461367   | 0.7353808474906784 |
| features_without_stacks | 0.9003355928075637   | 0.7354122095299801 |
| features_stack1         | 0.8914187473457081   | 0.7355281405860119 |
| features_stack2         | 0.8915824222847257   | 0.7352778823288092 |

### V. Outcome/Results

Our final model is a stacked ensemble, consisting of KNN Regression and Decision Tree Regression as stacks, and K-Means as our final model for predictions. To determine the star prediction for any new datapoint, we calculate the `stack_1` and `stack_2` predictions, after which we cluster the datapoint using our K-Means clustering algorithm. Based on the cluster that this new data point is placed in, we assign it a star rating (which has been pre-calculated per cluster using the K-Means model that was fitted on our training data).

Looking ahead, we may be able to further improve accuracy by reintroducing certain features that were dropped during the data preparation period. Features we considered engineering were:
- Cuisine (categorical) - would require us to categorize the ‘cuisine’ attribute that a portion of the data entries included (the rest had missing values). This would involve the use of a dictionary that classifies any nationality from those listed under this attribute (eg. Greek, Mexican) to a general region, such as Mediterranean or South Asian. Categorization would be necessary to reduce excessive dimensions in the case of one-hot-encoding.
- Review counts - could potentially weigh the “credibility” of a data point during model training based on its review count. However, this may introduce class imbalances and bais into the dataset, since most entries for review_count were < 10 with several outliers between 4,000-10,000. Additionally, it may be unfair to associate a newer restaurant (has a low review_count) with a lower rating.

### To Run Our Code

All of our final code is in `submission.ipnyb`, while all of our testing/exploration code can be found in the `workbooks/` directory. We use `data/yelp_academic_dataset_business.json` as the path to access the dataset, so please ensure that you upload the dataset with a matching directory structure to run the notebooks seamlessly.

Our chosen dataset is accessible here: https://yelp.com/dataset/download. We used the `yelp_academic_dataset_business.json` file.
