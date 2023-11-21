import numpy as np
import cv2

import cv2
import numpy as np


def add_salt_and_pepper_noise(image, amount=0.04):
    row, col, ch = image.shape
    s_vs_p = 0.5
    out = np.copy(image)

    num_salt = np.ceil(amount * row * col * s_vs_p).astype(int)

    x_coords = np.random.randint(0, row, num_salt)
    y_coords = np.random.randint(0, col, num_salt)
    out[x_coords, y_coords] = 1

    num_pepper = np.ceil(amount * row * col * (1. - s_vs_p)).astype(int)

    x_coords = np.random.randint(0, row, num_pepper)
    y_coords = np.random.randint(0, col, num_pepper)
    out[x_coords, y_coords] = 0

    return out


def add_gaussian_noise(image, mean=0, sigma=40):
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)


image = cv2.imread('input-pic.jpg')
output_size = (640, 480)

noisy_image_salt_pepper = add_salt_and_pepper_noise(image)

noisy_image_gaussian = add_gaussian_noise(image)

filter_size = 5

denoised_image_salt_pepper_median = cv2.medianBlur(noisy_image_salt_pepper, filter_size)
denoised_image_gaussian_median = cv2.medianBlur(noisy_image_gaussian, filter_size)

denoised_image_salt_pepper_gaussian = cv2.GaussianBlur(noisy_image_salt_pepper, (filter_size, filter_size), 0)
denoised_image_gaussian_gaussian = cv2.GaussianBlur(noisy_image_gaussian, (filter_size, filter_size), 0)

denoised_image_salt_pepper_dilate = cv2.dilate(noisy_image_salt_pepper, None, iterations=filter_size)
denoised_image_gaussian_dilate = cv2.dilate(noisy_image_gaussian, None, iterations=filter_size)

cv2.imshow('Original Image', cv2.resize(image, output_size))
cv2.imshow( 'Noisy Image (Salt and Pepper)', cv2.resize(noisy_image_salt_pepper, output_size))
cv2.imshow('Denoised Image (Salt and Pepper - Median)', cv2.resize(denoised_image_salt_pepper_median, output_size))
cv2.imshow('Denoised Image (Salt and Pepper - Gaussian)', cv2.resize(denoised_image_salt_pepper_gaussian, output_size))
cv2.imshow('Noisy Image (Gaussian)', cv2.resize(noisy_image_gaussian, output_size))
cv2.imshow('Denoised Image (Gaussian - Median)', cv2.resize(denoised_image_gaussian_median, output_size))
cv2.imshow('Denoised Image (Gaussian - Gaussian)', cv2.resize(denoised_image_gaussian_gaussian, output_size))
cv2.imshow('Denoised Image (Salt and Pepper - Dilation)', cv2.resize(denoised_image_salt_pepper_dilate, output_size))
cv2.imshow('Denoised Image (Gaussian - Dilation)', cv2.resize(denoised_image_gaussian_dilate, output_size))


cv2.imwrite('/home/reza/Desktop/opencv-noise/original_image.jpg', image)
cv2.imwrite('/home/reza/Desktop/opencv-noise/noisy_image_salt_pepper.jpg', noisy_image_salt_pepper)
cv2.imwrite('/home/reza/Desktop/opencv-noise/noisy_image_gaussian.jpg', noisy_image_gaussian)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_salt_pepper_median.jpg', denoised_image_salt_pepper_median)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_gaussian_median.jpg', denoised_image_gaussian_median)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_salt_pepper_gaussian.jpg', denoised_image_salt_pepper_gaussian)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_gaussian_gaussian.jpg', denoised_image_gaussian_gaussian)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_salt_pepper_dilate.jpg', denoised_image_salt_pepper_dilate)
cv2.imwrite('/home/reza/Desktop/opencv-noise/denoised_image_gaussian_dilate.jpg', denoised_image_gaussian_dilate)


cv2.waitKey(0)
cv2.destroyAllWindows()

