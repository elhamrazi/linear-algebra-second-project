import numpy as np
import PIL
from PIL import Image
from PIL.ExifTags import TAGS
# opening the image
image = Image.open("tnc_86887171.jpg")
size = image.size
print(image.format)
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

tmp = np.asarray([[[1, 2, 1], [4, 2, 13]], [[7, 8, 1], [8, 4, 5]]])


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
print("the mean vector is:")
print(mean_vector)


def b_matrix_cal(arr, mean):
    a = arr.shape
    for i in range(a[0]):
        for j in range(a[1]):
            temp = arr[i][j].astype(float) - mean
            arr[i][j] = temp


b_matrix_cal(matrix, mean_vector)
transpose_matrix1 = matrix.transpose(2, 0, 1).reshape(3, -1)
transpose_matrix2 = transpose_matrix1.transpose()
covariance_matrix = np.dot(transpose_matrix1, transpose_matrix2)
print("the covariance matrix is:")
print((1/(size[0]*size[1]-1)) * covariance_matrix)




