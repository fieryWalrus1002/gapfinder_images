# Next steps:

1. roi image width should be standardized to about 60 pixels

- run 7b to re-extract, and set to the appropriate output

2. proceed with 8. don't slice, instead just use the whole image in the histogram
3. the peak widths should be calculated at about 50% of the peak height

- this is a good way to standardize the width of the peaks, and it's a good way to compare the peaks to each other

Done re-extacting. Now I will proceed with 8.


# 9-26 Update

## Current Status
Image have been re-extracted. 

## Next steps:
Proceed with 9, and calculate the peak widths at about 50% of the peak height.

Use the values for repeat distance from Vaclav to compare against delta peak center values.


# For next week:
- Calculate the nm/pixel ratio for the images
  - go through the raw images. 
  - Isolate the bottom of the image, where the black border is.
  - This border contains a scale bar in white, and white text that says "200 nm" or "500 nm"
  - Use OCR to extract the scale, and then get a contour of the scale bar
  - The scale bar will be a varying amount of pixels
  - We can then use this to calculate the nm/pixel ratio, and then use this to calculate the nm/pixel ratio for the entire image
  - Export this number to a csv file, and then use this to calculate the nm/pixel ratio for the entire image
    - Should have the raw image name, the nm/pixel ratio, the scale value, the scale bar length in pixels, and the scale bar length in nm to verify it is correct
- Calibrate the extracted values from the histograms (9_signal_processing.ipynb) using the nm/pixel ratio
  - This will allow us to get the nm values for the extracted peak widths
  - This will allow us to compare the extracted peaks to the delta peak center values

# Output desired from the above:

Two ouput types:



### Type 1:
```
013 WT-DA-1 19k new sec strip 2
Number of peaks: 8
Average distance between peaks: 13.57
Average peak height: 15810.00
Average peak width: 0.00
Peak_num 0: peak: 5, peak_height: 15810, width: 5.27, left base: 2.18, right base: 7.45, center: 4.82
Peak_num 1: peak: 18, peak_height: 15810, width: 5.16, left base: 15.26, right base: 20.43, center: 17.84
Peak_num 2: peak: 31, peak_height: 15810, width: 4.82, left base: 28.89, right base: 33.70, center: 31.30
Peak_num 3: peak: 44, peak_height: 15810, width: 5.56, left base: 42.00, right base: 47.56, center: 44.78
Peak_num 4: peak: 59, peak_height: 15810, width: 5.00, left base: 56.69, right base: 61.69, center: 59.19
Peak_num 5: peak: 72, peak_height: 15810, width: 4.95, left base: 69.72, right base: 74.67, center: 72.20
Peak_num 6: peak: 86, peak_height: 15810, width: 5.09, left base: 83.22, right base: 88.32, center: 85.77
Peak_num 7: peak: 100, peak_height: 15810, width: 4.81, left base: 97.53, right base: 102.34, center: 99.94
```
Note: calculate the peak value using the left and right base values, so you get a continuous value for the peak instead of the integer value (which is an artifact of the histogram extraction using the index of the peak)

## Type 2: derived from the first type

```
Peak-to-Peak, Trough-to-Trough
Data that you van put on an excel and average:
Distance between maxima, distance between minima, and FWHM for both maxima and minima. Number of maxima. 

7 maxima-to-maxima, 6 minima-to-minima, 8 maxima FWHM, 7 minima FWHM for 8 peaks.

```