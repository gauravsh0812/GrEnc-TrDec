# base code has borrowed from https://www.geeksforgeeks.org/python-opencv-connected-component-labeling-and-analysis/#

import os
import cv2
import numpy as np
import multiprocessing as mp
import yaml


with open('configs/config.yaml') as f:
	cfg = yaml.safe_load(f)
	

def sortXY(Xs, Ys, pts, rgns):
	
	sorted_Xs = sorted(Xs)
	sorted_coord = list()
	sorted_pts = list()
	for i,x in enumerate(sorted_Xs):
		idx = Xs.index(x)
		sorted_coord.append((x,Ys[idx]))
		sorted_pts.append(pts[idx])
		cv2.imwrite(f"regions/{i}.png", rgns[idx])

	return sorted_coord, sorted_pts


def main_parallel(im, display=False):

	print(im)
	img = cv2.imread(f"data/images/{im}")

	# preprocess the image
	if img.shape[2] == 3:
		gray_img = cv2.cvtColor(img ,
								cv2.COLOR_BGR2GRAY)
	else:
		gray_img = img

	# Applying 7x7 Gaussian Blur
	# blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
	# blurred = gray_img

	# Applying threshold
	threshold = cv2.threshold(gray_img, 0, 255,
		cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

	# Apply the Component analysis function
	analysis = cv2.connectedComponentsWithStats(threshold,
												4,
												cv2.CV_32S)
	(totalLabels, label_ids, values, centroid) = analysis

	# Initialize a new image to
	# store all the output components
	output = np.zeros(gray_img.shape, dtype="uint8")

	# Loop through each component
	Xs, Ys = [], []
	pts = []  # list[list[tuple]]
	rgns = []

	for i in range(1, totalLabels):
		# Area of the component
		area = values[i, cv2.CC_STAT_AREA]
		if (area > 0):# and (area < 500):
			# Create a new image for bounding boxes
			new_img=img.copy()

			# Now extract the coordinate points
			x1 = values[i, cv2.CC_STAT_LEFT]
			y1 = values[i, cv2.CC_STAT_TOP]
			w = values[i, cv2.CC_STAT_WIDTH]
			h = values[i, cv2.CC_STAT_HEIGHT]

			# Coordinate of the bounding box
			pt1 = (x1, y1)
			pt2 = (x1+ w, y1+ h)
			(X, Y) = centroid[i]

			# appending in dictionary
			Xs.append(X)
			Ys.append(Y)
			pts.append([pt1, pt2])
			# print(f"{X}, {Y} \t {pt1} \t {pt2}")

			# Bounding boxes for each component
			cv2.rectangle(new_img,pt1,pt2,
						(0, 255, 0), 3)
			cv2.circle(new_img, (int(X),
								int(Y)),
					4, (0, 0, 255), -1)

			# Create a new array to show individual component
			component = np.zeros(gray_img.shape, dtype="uint8")
			componentMask = (label_ids == i).astype("uint8") * 255

			# Apply the mask using the bitwise operator
			component = cv2.bitwise_or(component,componentMask)
			output = cv2.bitwise_or(output, componentMask)

			rgns.append(component)

			# Show the final images
			if display:
				cv2.imshow("Image", new_img)
				cv2.imshow("Individual Component", component)
				cv2.imshow("Filtered Components", output)
				cv2.waitKey(1000)

			# writing the img
			cv2.imwrite(f'filtered_images/{im}', output)

	# sorting the coord based on the x-coord
	sorted_coord, sorted_pts = sortXY(Xs, Ys, pts, rgns)
	y=0
	for i,s in enumerate(sorted_coord):
		y+=s[1]
		print(f"{i} -- {s}")
	print(y/len(sorted_coord))


def main():

    args = cfg["building_graph"]

	# Loading the image
    # imgs = [im for im in os.listdir("./images") if ".png" in im]
    imgs = ["10.png"]

	# creating folders
    for _path in ["./filtered_images",
			   "./regions"]:
        if not os.path.exists(_path):
            os.mkdir(_path)
	
    with mp.Pool(args["ncpus"]) as pool:
        result = pool.map(main_parallel, imgs)

if __name__ == "__main__":
	main()