import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog

print("Select the original image:")
orig_path = filedialog.askopenfilename(title="Select the Original form image")
print("Select the template image:")
temp_path = filedialog.askopenfilename(title="Select the Template image of the form")


img_orig = cv2.imread(orig_path)
img_template = cv2.imread(temp_path)
if img_orig is  None or img_template is None:
    print("Error: Could not load the image or template. Please check the file paths.")
    exit()
img_orig_color = img_orig.copy()

img_orig_gray = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
img_template_gray = cv2.cvtColor(img_template, cv2.COLOR_BGR2GRAY)
#orig img
plt.figure(figsize=(12, 6))
plt.title("Original Image")
plt.imshow(img_orig_gray, cmap='gray')  
#template
plt.figure(figsize=(8, 5))
plt.imshow(img_template_gray, cmap='gray')
plt.axis('off')
#temp dimen
w, h = img_template_gray.shape[::-1]
print("Template size :", w, "x", h)

 #template matching
match_result = cv2.matchTemplate(img_orig_gray, img_template_gray, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(match_result)
#From max_loc[1] (top-left y-coordinate) to max_loc[1] + h (bottom-right y-coordinate).
#From max_loc[0] (top-left x-coordinate) to max_loc[0] + w (bottom-right x-coordinate).
match_region = img_orig_color[max_loc[1]:max_loc[1]+h, max_loc[0]:max_loc[0]+w]
output_path = "C:/Users/RUCHI/Documents/riya/cropped_template.png"
cv2.imwrite(output_path, match_region)
print(f"Cropped template saved at : {output_path}")
plt.show()
