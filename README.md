# Predicting the remaining useful life of a machine  
Remaining useful life (RUL), as the name suggests, is the amount of time left with a machine before it requires replacement or breakdown mitigation.  
The problem at hand is to come up with a machine learning model to predict the RUL (measured in operation cycles) based on time-series data of sensor measurements typically available from aircraft gas turbine 
engines.  
The [dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan) used here is the Turbofan Engine Degradation Simulation Data Set created by NASA. 
It essentially is a time series of some sensor readings collected over a fleet of engines. 
For details on the format of the dataset, check readme in the CMAPSS_Data folder. 
