# NYS Satellite Chlorphyll-A model

### By [Jillian Greene](https://github.com/jilliangreene), [Lenny Metlitsky](https://github.com/leomet07), Andrew Levine, [Erin Foley](https://github.com/erinf26), [Melody Henry](https://github.com/melodyghenry) Marzi Azarderkhsh, Reginald Blake, and Hamid Norouzi

## Abstract


> Algal bloom proliferation across the United States has been increasing in congruence with several anthropogenically influenced processes. Challenges in monitoring algal bloom growth include the high labor and equipment requirements necessary to quantify algal presence which makes widespread availability of data limited. In New York, USA, where numerous inland lakes experience severe anthropogenic impacts, only a fraction of the over 7,000 lakes have reliable quantitative algal bloom data. Operational remote sensing can be used to fill the gap in unmonitored lakes based on the relationship between surface reflectance and in-situ chlorophyll-a concentrations. Previous remote sensing techniques are challenging to apply to small lakes due to spatial resolution capabilities, atmospheric correction algorithms, and lack of a comprehensive empirical formula. In this study, we used Landsat-8 and -9, (2013-2023), and Sentinel-2 (2019-2023) reflectance values coupled with key watershed characteristics (National Land Cover Database percentages and lake morphology) to model algal presence in New York lakes. Images were processed using the Modified Atmospheric correction for INland waters (MAIN) tailored toward dark images features. We implement several nonlinear machine learning models, including Support Vector Regression (SVR), Random Forest Regression (RFR), gradient-boosted regression (GBR), and Extra Trees Regression (ETR). The best performing model, Extra Trees Regression, was found to qualitatively estimate chlorophyll-a presence with marginal model error (R2 = 0.72, RMSE = 8.19 ug/l). From these results, we can obtain a comprehensive view of algal blooms across New York from 2013 - the present, which will inform stakeholders of the trends in both presence and concentrations as well as the time frame of both. The model covers >450 lakes in New York with high resolution imagery to include the small inland lakes that experience the highest nutrient concentrations and biological activity. Results from this study will be made available on a public website application that will contain in situ data, key lake characteristics, model time-series, and model raster predictions. This model procedure is easily replicable for expansion outside of New York State and can be expanded to include more lakes in other regions.


## Structure

This is a monorepo containing three subdirectories. They represent our workflow, with the order they are described in below. However, each subproject can still be run independently at any time.

1. ``satellite_fetch/``: First, images are fetched from either the Landsat 8/9 or the Sentinel 2a/2b satellites for a specific NYS lake. Images undergo [MAIN atmospheric correction](https://github.com/Nateme16/geo-aquawatch-water-quality/tree/main/Atmospheric%20corrections), and then they are masked to the shape of the lake, and images with high cloud coverage are discarded. Scripts to download all images for a specific interval (or entire year) of a given set of lakes (including all 4k lakes) are provided as well.

2. ``ml_model/``: This is where the training data is assembled, split into the training and testing subsets, and passed into a [ExtraTreesRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html). After analyzing the performance of the model (with figures), entire directories of images downloaded from step 1 can be passed in to run en masse. Each image, along with some metadata about its time, corresponding lake, and filepaths, is saved as a new record in a MongoDB collection for all spatial predictions.

3. ``analysis/``: This is where the production runs from step 2 are analyzed. It queries the MongoDB database for any images matching analysis criteria. If they are found, the output rasters are opened and analyzed for their min, max, mean, STD. It is also possible to target specific points within the output raster based of the latitude and longitude. The many analysis scripts are detailed in the analysis README. One particularly useful script analyzes the models RMSE/MAE within 60m circles about centroids (usually 2-3 pixels radius about the centroid). The main script analyzes and plots the mean chla since 2013 of a specific lake or all lakes.

## Acknowledgments

### Funding:

This research was supported by funding from the NASA Climate Change Research Initiative (CCRI) and the Department of Energy (DOE) under award number DE-SC0023208, as well as by the Center for Remote Sensing and Earth System Sciences (ReSESS) at the New York City College of Technology. 
