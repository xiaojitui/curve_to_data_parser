# curve_to_data_parser

The script is used to detect curves in a plot, then parse curves into numerical data points. 

<br><br>
The algorithm is:

(1) detect X-axis and Y-axis in the plot

(2) detect and parse axis labels*

(3) detect curves and parse curves into data points

*notes: to parse axis labels, 'pytesseract' package and Google Tesseract-Ocr Engine need to be installed. 


To find curves in a plot and convert them to data points, run: 

python curveparser.py

The parsed data points will be saved in "curve_data_points.pkl". The format is: {curve_id: [[xi, yi]]}


