from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import cv2


def euclidean_distance_plot(point):
    dist_1 = sqrt((point[0] - 255)**2+point[1]**2+point[2]**2)
    dist_2 = sqrt((point[1] - 255)**2+point[0]**2+point[2]**2)
    dist_3 = sqrt((point[2] - 255)**2+point[0]**2+point[1]**2)
    distance_array = [abs(dist_1), abs(dist_2), abs(dist_3)]
    return distance_array.index(min(distance_array))

def problem_2():
    img = cv2.imread('test.png')
    row_index = -1
    col_index = -1
    for i in img:
        row_index += 1
        col_index = 0
        for j in i:
            result = euclidean_distance_plot(j)
            if result == 0:
                img[row_index][col_index] = [255, 0, 0]
            if result == 1:
                img[row_index][col_index] = [0, 255,0]
            if result == 2:
                img[row_index][col_index] = [0, 0,255]
            col_index+= 1

    cv2.imshow("Result",img)
    cv2.imwrite("Result.png",img)

def problem_1():
    word_array = []
    histogram = {}
    histogram_numpy_normalize = np.empty((26,1))

    for x in range(97,123):
        histogram[chr(x)] = 0

    with open("test.txt", "r") as f:
        for name in f:
            word_array+= name

    for word in word_array:
        asc_alpha = ord(word.lower())
        if asc_alpha > 96:
            if asc_alpha < 123:
                histogram[word.lower()] +=1


    histogram_numpy = np.array(list(histogram.values()))
    print(histogram)
    print(histogram_numpy)
    histogram_numpy_normalize = np.array([float(i)/sum(histogram_numpy) for i in histogram_numpy])
    print(histogram_numpy_normalize)
    plt.bar(histogram.keys(), histogram.values(), color = 'red', width = 0.5)
    plt.show()
    plt.bar(np.arange(len(histogram_numpy_normalize)),histogram_numpy_normalize, color='magenta', width=0.5)
    plt.show()


def problem_3():
    A = np.array([[3,3],[4,2]])
    B = np.array([11.25,10])

    X = np.linalg.inv(A).dot(B)
    print(X)

problem_1()
problem_2()
problem_3()