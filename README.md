# bpm_from_camera
Takes in 1min recording of finger on iphone camera lens and estimates heart rate and HRV using photoplethysmography


For each frame, convert RGB to HSV (Hue-Saturation-Value). Normalized hue signal over time reflects pulse. Distance between peaks yields beats per minute, variability in these distances yields HRV.

https://www.researchgate.net/publication/316892007_Real-time_Heart_Rate_Measurement_based_on_Photoplethysmography_using_Android_Smartphone_Camera

