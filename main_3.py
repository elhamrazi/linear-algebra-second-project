import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def decompose_2d_matrix(mat):
    new_r = [[0] * (shapes[1]) for i in range((shapes[0]))]
    new_g = [[0] * (shapes[1]) for i in range((shapes[0]))]
    new_b = [[0] * (shapes[1]) for i in range((shapes[0]))]
    for i in range(0, len(mat[0])):
        new_r[int(i / shapes[1])][i % shapes[1]] = mat[0][i]
    for i in range(0, len(mat[0])):
        new_g[int(i / shapes[1])][i % shapes[1]] = mat[1][i]
    for i in range(0, len(mat[0])):
        new_b[int(i / shapes[1])][i % shapes[1]] = mat[2][i]
    return new_r, new_g, new_b


def combine_matrix():
    compressed = (np.dstack((newr, newg, newb)))
    return compressed


# opening the image
image = Image.open("tnc_86887171.jpg")

# save the image in a numpy 3D array
matrix = np.array(image)
shapes = matrix.shape
print(matrix)

# getting the metadata, if any:
exifdata = image.getexif()
for tag_id in exifdata:
    # get the tag name, instead of human unreadable tag id
    tag = TAGS.get(tag_id, tag_id)
    data = exifdata.get(tag_id)
    # decode bytes
    if isinstance(data, bytes):
        data = data.decode()
    print(f"{tag:25}: {data}")


# we need to save the 3d data as a 2d to work with it
reshaped_matrix = matrix.transpose(2, 0, 1).reshape(3, -1)
b_matrix = np.copy(reshaped_matrix)
print(reshaped_matrix.shape)
e = b_matrix.transpose()

# calculating the mean vector
mean_vector = np.sum(reshaped_matrix, axis=1)
mean_vector = mean_vector / (shapes[0] * shapes[1])
print("the mean vector is:", mean_vector)
# print(b_matrix)
# making the b_matrix for covariance matrix

ee = e - mean_vector
print(ee.transpose())

# getting the covariance matrix
b = ee.transpose()
bt = ee
covariance_matrix = np.dot(b, bt)
covariance_matrix = (1/(shapes[0] * shapes[1] - 1)) * covariance_matrix
print("the covariance matrix is:")
print(covariance_matrix)

# getting the eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
eigenvalues_sorted = np.sort(eigenvalues)
eigenvectors_sorted = eigenvectors[:, eigenvalues_sorted.argsort()]
print("the eigenvalues and the eigenvectors of the covariance matrix:")
print(eigenvalues_sorted)
print(eigenvectors_sorted)

eigvectors = np.asarray([[eigenvectors_sorted[0][2], eigenvectors_sorted[0][1], eigenvectors_sorted[0][0]],
                        [eigenvectors_sorted[1][2], eigenvectors_sorted[1][1], eigenvectors_sorted[1][0]],
                        [eigenvectors_sorted[2][2], eigenvectors_sorted[2][1], eigenvectors_sorted[2][0]]])
print(eigvectors)
y = np.dot(eigvectors.transpose(), b)
newr, newg, newb = decompose_2d_matrix(y)
plt.imshow(combine_matrix())
plt.show()
print("the new matrix for the image matrix:")
print(y)


# calculating the total variance
tr = sum(eigenvalues_sorted)
print("the total variance is: ", tr)

components = [eigenvalues_sorted[2] / tr * 100, eigenvalues_sorted[1] / tr * 100, eigenvalues_sorted[0] / tr * 100]
print(*components)

pca = []
for i in range(3):
    if components[i] < 5:
        pca.append(i)
# print(pca)

# reducing the dimension
for i in pca:
    y[i] = 0
if len(pca) == 0:
    print("we cannot reduce the dimension.")

print(y)
x = y.transpose()
new_y = x.reshape(2000, 2000, 3)
print(np.round(np.absolute(new_y)))
print(new_y.shape)

