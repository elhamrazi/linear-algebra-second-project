import numpy as np
import PIL
from PIL import Image
from PIL.ExifTags import TAGS
# opening the image
image = Image.open("tnc_86887171.jpg")
print(image.format)
print(image.size)
print(image.mode)
# saving it as a numpy array. it is three-dimensional
matrix = np.array(image)
print(type(matrix))
print(matrix.shape)
# getting the metadata of the image


exifdata = image.getexif()
for tag_id in exifdata:
    # get the tag name, instead of human unreadable tag id
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    # decode bytes
    if isinstance(data, bytes):
        data = data.decode()
    print(f"{tag:25}: {data}")

# function to calculate the mean
print(matrix[0][0])

def mean_cal(arr):
    sum = [0, 0, 0]
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            sum += arr[i][j]
    x = sum / (a[0] * a[1])
    return x


# calculating the covariance matrix
mean_vector = mean_cal(matrix)
print(mean_vector)


def covariance_cal(arr, mean):
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            temp = arr[i][j].astype(float) - mean
            arr[i][j] = temp


covariance_cal(matrix, mean_vector)
print(matrix[0][0])
transpose_matrix = matrix.transpose()
print(transpose_matrix[0][0])




