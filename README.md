# taxi_fare_prediction
Supervised ML project to predict final fare of a taxi ride using various features of the journey.

Data Files
The dataset is composed of the following files:

train.csv: The training set, which includes the target variable 'total_amount' and accompanying feature attributes.

test.csv: The test set, containing similar feature attributes but without the target variable 'total_amount,' as it is the variable to be predicted.

sample_submission.csv: A sample submission file provided in the correct format for competition submissions.

Columns Description

The dataset comprises various columns, each offering valuable insights into taxi rides. Notably:

total_amount: The total amount paid by the traveler for the taxi ride.

VendorID: An identifier for taxi vendors.

tpep_pickup_datetime and tpep_dropoff_datetime: Timestamps indicating pickup and dropoff times.

passenger_count: The number of passengers during the ride.

trip_distance: The distance traveled during the trip.

RatecodeID: Rate code for the ride.

store_and_fwd_flag: A flag indicating whether the trip data was stored and forwarded.

payment_type: Payment type used for the ride.