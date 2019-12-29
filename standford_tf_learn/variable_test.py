import tensorflow as tf
import os
import tensorflow.contrib.eager as tfe
import numpy as np
from PIL import Image

tf.enable_eager_execution()

# def get_image(img_path, img_width=90, img_height=90, channel_num=3):
#     try:
#         img = Image.open(img_path)
#         num_bands = img.getbands()
#         if len(num_bands) == 4:  # png format
#             background = Image.new("RGB", img.size, (255, 255, 255))
#             background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
#             img = background
#         img = img.resize((img_width, img_height))
#         pixes = np.zeros((img_width, img_height, channel_num), np.float32)
#         pix = img.load()
#         for x in range(img_width):
#             for y in range(img_height):
#                 pixes[x][y] = pix[x, y]
#         return pixes
#     except IOError:
#         print("image broken")
#
#
# def get_test_batch(test_file_path):
#     img_list = []
#     for i, img_name in enumerate(os.listdir(test_file_path)):
#         img_path = test_file_path + img_name
#         print(img_path)
#         image_pixes = get_image(img_path)
#         image_pixes = tf.expand_dims(image_pixes, axis=0)
#         img_list.append(image_pixes)
#     return tf.concat(img_list, axis=0)

# def record_img_file_path(img_file_path):
# with open('ready_no_edit_file_path', 'w') as file:
#     for path in os.listdir("D:/test_img/"):
#         file.write("D:/test_img/" + path)
#         file.write("\n")
# file.close()

filename = "D:/test_img/569dec23N1a8dc992.png"
image_string = tf.read_file(filename)
img = tf.cond(
    tf.image.is_jpeg(image_string),
    lambda: tf.image.decode_jpeg(image_string, channels=3),
    lambda: tf.image.decode_png(image_string, channels=3))
print(img)

# def _parse_function(filename):
#     image_string = tf.read_file(filename)
#     print(image_string)
#     image_decoded = tf.cond(
#         tf.image.is_jpeg(image_string),
#         lambda: tf.image.decode_jpeg(image_string, channels=3),
#         lambda: tf.image.decode_png(image_string, channels=3))
#     # image_decoded.set_shape([None, None, 3])
#     image_resized = tf.image.resize_images(image_decoded, [90, 90])
#     return image_resized
#
# sess = tf.Session()
# test_file_path  = "/export/sdb/home/zhaijianwei/choose_white_img/no_edit_img_file/"
# filenames = [test_file_path + img_name for img_name in os.listdir(test_file_path)]
# for filename in filenames:
#     print(filename)
#     sess.run(_parse_function(filename))

# image_string = tf.read_file(filename)
# if filename.split(".")[-1] == "png":
#     image_decoded = tf.image.decode_png(filename, channels=3)
# else:
#     image_decoded = tf.image.decode_jpeg(image_string, channels=3)
#
# # image_decoded.set_shape([None, None, 3])
# image_resized = tf.image.resize_images(image_decoded, [90, 90])
#
# sess = tf.Session()
# print(sess.run(img))

# sess = tf.Session()
# batch = get_test_batch("D:/test_img/")
# print(sess.run(tf.shape(batch)))


def get_filter_img_path(img_path):
    filter_img_names = os.listdir(img_path)

    file_reader = open("/export/sdb/home/zhaijianwei/choose_white_img/no_edit_img_path_file", "r")
    file_writer = open("img_greater_250k_path", "w")

    img_pathes = file_reader.readlines()

    for file_path in img_pathes:
        img_name = file_path.split('/')[-1]
        if img_name in filter_img_names:
            file_writer.write(file_path)
            file_writer.write('\n')


get_filter_img_path("/export/sdb/home/zhaijianwei/choose_white_img/img_greater_250k")


