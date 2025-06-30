## Landsat8/9 and Sentinel2a/b Satellite Fetching
Images recieve MAIN correction as well.

## Installation

1. Make a virtual environment

```bash
python -m venv venv
```

2. Activate virtual environment

```bash
source venv/bin/activate
```

3. Install requirements

```bash
pip install -r requirements.txt
```


## Examples

**Example to download the first image without cloud cover in an interval with a resolution of 30mby30m per pixel:**
```bash
python functions.py out <your-google-earth-engine-project-name> 16364 "2019-08-01" "2019-08-24" 30 congers.tif
```

This downloads out/congers.tif of 2019-08-01, because that is the first available image within that interval.

**Example to download the first images within a 5 day subinterval of a larger interval with a resolution of 20mby20m per pixel:**
```bash
python multiple_dates.py out <your-google-earth-engine-project-name> 16364 "2019-06-01" "2019-08-31" 20 5 congers
```

This downloads ``out_mult_high_rescongers2019-06-11to2019-06-16.tif`` and then ``out_mult_high_res/congers2019-06-21to2019-06-26.tif`` etc.. all the way up to  ``out_mult_high_res/congers2019-08-25to2019-08-30.tif``

We used an interval of 5 because sentinel 2a and 2b have a visit frequency of every 10 days each, offset by 5 days, so there is a visit by a sentinel 2 satellite every 5 days.

###  This was originally from the [leomet07/PredictChlorophyllALibrary](https://github.com/leomet07/PredictChlorophyllALibrary)
