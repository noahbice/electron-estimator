# electron-estimator
An application to provide guidance in selecting an appropriate energy and bolus thickness for electron treatments based on PDDs. Enter target minimum and maximum depth and review energy/bolus combinations based on entrance dose, exit dose, hot spot, and depth dose. The app is available at: https://share.streamlit.io/noahbice/electron-estimator/main/main.py.

Higher-than-necessary energies are penalized based on some input ``OAR Depth'' and target OAR dose, stated in terms of the prescription dose. The default OAR depth is 1.5 times the maximum target depth, and the default OAR target 

The following objective function is used to sort possible combinations:
w_1	|target entrance dose – 100 | + w_2	|target exit dose – 100 | + w_3 (hotspot dose – 100) + w_4 (max(90, skin dose) – 90) + w_5 (max(OAR target dose, true OAR dose) – OAR target dose)
