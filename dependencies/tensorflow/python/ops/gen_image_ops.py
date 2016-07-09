"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def adjust_contrast(images, contrast_factor, min_value, max_value, name=None):
  r"""Deprecated. Disallowed in GraphDef version >= 2.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
    contrast_factor: A `Tensor` of type `float32`.
    min_value: A `Tensor` of type `float32`.
    max_value: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return _op_def_lib.apply_op("AdjustContrast", images=images,
                              contrast_factor=contrast_factor,
                              min_value=min_value, max_value=max_value,
                              name=name)


def _adjust_contrastv2(images, contrast_factor, name=None):
  r"""Adjust the contrast of one or more images.

  `images` is a tensor of at least 3 dimensions.  The last 3 dimensions are
  interpreted as `[height, width, channels]`.  The other dimensions only
  represent a collection of images, such as `[batch, height, width, channels].`

  Contrast is adjusted independently for each channel of each image.

  For each channel, the Op first computes the mean of the image pixels in the
  channel and then adjusts each component of each pixel to
  `(x - mean) * contrast_factor + mean`.

  Args:
    images: A `Tensor` of type `float32`. Images to adjust.  At least 3-D.
    contrast_factor: A `Tensor` of type `float32`.
      A float multiplier for adjusting contrast.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. The contrast-adjusted image or images.
  """
  return _op_def_lib.apply_op("AdjustContrastv2", images=images,
                              contrast_factor=contrast_factor, name=name)


def decode_jpeg(contents, channels=None, ratio=None, fancy_upscaling=None,
                try_recover_truncated=None, acceptable_fraction=None,
                name=None):
  r"""Decode a JPEG-encoded image to a uint8 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the JPEG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.

  If needed, the JPEG-encoded image is transformed to match the requested number
  of color channels.

  The attr `ratio` allows downscaling the image by an integer factor during
  decoding.  Allowed values are: 1, 2, 4, and 8.  This is much faster than
  downscaling the image later.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The JPEG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    ratio: An optional `int`. Defaults to `1`. Downscaling ratio.
    fancy_upscaling: An optional `bool`. Defaults to `True`.
      If true use a slower but nicer upscaling of the
      chroma planes (yuv420/422 only).
    try_recover_truncated: An optional `bool`. Defaults to `False`.
      If true try to recover an image from truncated input.
    acceptable_fraction: An optional `float`. Defaults to `1`.
      The minimum required fraction of lines before a truncated
      input is accepted.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `uint8`. 3-D with shape `[height, width, channels]`..
  """
  return _op_def_lib.apply_op("DecodeJpeg", contents=contents,
                              channels=channels, ratio=ratio,
                              fancy_upscaling=fancy_upscaling,
                              try_recover_truncated=try_recover_truncated,
                              acceptable_fraction=acceptable_fraction,
                              name=name)


def decode_png(contents, channels=None, dtype=None, name=None):
  r"""Decode a PNG-encoded image to a uint8 or uint16 tensor.

  The attr `channels` indicates the desired number of color channels for the
  decoded image.

  Accepted values are:

  *   0: Use the number of channels in the PNG-encoded image.
  *   1: output a grayscale image.
  *   3: output an RGB image.
  *   4: output an RGBA image.

  If needed, the PNG-encoded image is transformed to match the requested number
  of color channels.

  Args:
    contents: A `Tensor` of type `string`. 0-D.  The PNG-encoded image.
    channels: An optional `int`. Defaults to `0`.
      Number of color channels for the decoded image.
    dtype: An optional `tf.DType` from: `tf.uint8, tf.uint16`. Defaults to `tf.uint8`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `dtype`. 3-D with shape `[height, width, channels]`.
  """
  return _op_def_lib.apply_op("DecodePng", contents=contents,
                              channels=channels, dtype=dtype, name=name)


def draw_bounding_boxes(images, boxes, name=None):
  r"""Draw bounding boxes on a batch of images.

  Outputs a copy of `images` but draws on top of the pixels zero or more bounding
  boxes specified by the locations in `boxes`. The coordinates of the each
  bounding box in `boxes are encoded as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example, if an image is 100 x 200 pixels and the bounding box is
  `[0.1, 0.5, 0.2, 0.9]`, the bottom-left and upper-right coordinates of the
  bounding box will be `(10, 40)` to `(50, 180)`.

  Parts of the bounding box may fall outside the image.

  Args:
    images: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, depth]`. A batch of images.
    boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, num_bounding_boxes, 4]` containing bounding
      boxes.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
    4-D with the same shape as `images`. The batch of input images with
    bounding boxes drawn on the images.
  """
  return _op_def_lib.apply_op("DrawBoundingBoxes", images=images, boxes=boxes,
                              name=name)


def encode_jpeg(image, format=None, quality=None, progressive=None,
                optimize_size=None, chroma_downsampling=None,
                density_unit=None, x_density=None, y_density=None,
                xmp_metadata=None, name=None):
  r"""JPEG-encode an image.

  `image` is a 3-D uint8 Tensor of shape `[height, width, channels]`.

  The attr `format` can be used to override the color format of the encoded
  output.  Values can be:

  *   `''`: Use a default format based on the number of channels in the image.
  *   `grayscale`: Output a grayscale JPEG image.  The `channels` dimension
      of `image` must be 1.
  *   `rgb`: Output an RGB JPEG image. The `channels` dimension
      of `image` must be 3.

  If `format` is not specified or is the empty string, a default format is picked
  in function of the number of channels in `image`:

  *   1: Output a grayscale image.
  *   3: Output an RGB image.

  Args:
    image: A `Tensor` of type `uint8`.
      3-D with shape `[height, width, channels]`.
    format: An optional `string` from: `"", "grayscale", "rgb"`. Defaults to `""`.
      Per pixel image format.
    quality: An optional `int`. Defaults to `95`.
      Quality of the compression from 0 to 100 (higher is better and slower).
    progressive: An optional `bool`. Defaults to `False`.
      If True, create a JPEG that loads progressively (coarse to fine).
    optimize_size: An optional `bool`. Defaults to `False`.
      If True, spend CPU/RAM to reduce size with no quality change.
    chroma_downsampling: An optional `bool`. Defaults to `True`.
      See http://en.wikipedia.org/wiki/Chroma_subsampling.
    density_unit: An optional `string` from: `"in", "cm"`. Defaults to `"in"`.
      Unit used to specify `x_density` and `y_density`:
      pixels per inch (`'in'`) or centimeter (`'cm'`).
    x_density: An optional `int`. Defaults to `300`.
      Horizontal pixels per density unit.
    y_density: An optional `int`. Defaults to `300`.
      Vertical pixels per density unit.
    xmp_metadata: An optional `string`. Defaults to `""`.
      If not empty, embed this XMP metadata in the image header.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. 0-D. JPEG-encoded image.
  """
  return _op_def_lib.apply_op("EncodeJpeg", image=image, format=format,
                              quality=quality, progressive=progressive,
                              optimize_size=optimize_size,
                              chroma_downsampling=chroma_downsampling,
                              density_unit=density_unit, x_density=x_density,
                              y_density=y_density, xmp_metadata=xmp_metadata,
                              name=name)


def encode_png(image, compression=None, name=None):
  r"""PNG-encode an image.

  `image` is a 3-D uint8 or uint16 Tensor of shape `[height, width, channels]`
  where `channels` is:

  *   1: for grayscale.
  *   3: for RGB.
  *   4: for RGBA.

  The ZLIB compression level, `compression`, can be -1 for the PNG-encoder
  default or a value from 0 to 9.  9 is the highest compression level, generating
  the smallest output, but is slower.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `uint16`.
      3-D with shape `[height, width, channels]`.
    compression: An optional `int`. Defaults to `-1`. Compression level.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`. 0-D. PNG-encoded image.
  """
  return _op_def_lib.apply_op("EncodePng", image=image,
                              compression=compression, name=name)


def hsv_to_rgb(images, name=None):
  r"""Convert one or more images from HSV to RGB.

  Outputs a tensor of the same shape as the `images` tensor, containing the RGB
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  See `rgb_to_hsv` for a description of the HSV encoding.

  Args:
    images: A `Tensor` of type `float32`.
      1-D or higher rank. HSV data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. `images` converted to RGB.
  """
  return _op_def_lib.apply_op("HSVToRGB", images=images, name=name)


def rgb_to_hsv(images, name=None):
  r"""Converts one or more images from RGB to HSV.

  Outputs a tensor of the same shape as the `images` tensor, containing the HSV
  value of the pixels. The output is only well defined if the value in `images`
  are in `[0,1]`.

  `output[..., 0]` contains hue, `output[..., 1]` contains saturation, and
  `output[..., 2]` contains value. All HSV values are in `[0,1]`. A hue of 0
  corresponds to pure red, hue 1/3 is pure green, and 2/3 is pure blue.

  Args:
    images: A `Tensor` of type `float32`.
      1-D or higher rank. RGB data to convert. Last dimension must be size 3.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. `images` converted to HSV.
  """
  return _op_def_lib.apply_op("RGBToHSV", images=images, name=name)


def _random_crop(image, size, seed=None, seed2=None, name=None):
  r"""Randomly crop `image`.

  `size` is a 1-D int64 tensor with 2 elements representing the crop height and
  width.  The values must be non negative.

  This Op picks a random location in `image` and crops a `height` by `width`
  rectangle from that location.  The random location is picked so the cropped
  area will fit inside the original image.

  Args:
    image: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      3-D of shape `[height, width, channels]`.
    size: A `Tensor` of type `int64`.
      1-D of length 2 containing: `crop_height`, `crop_width`..
    seed: An optional `int`. Defaults to `0`.
      If either seed or seed2 are set to be non-zero, the random number
      generator is seeded by the given seed.  Otherwise, it is seeded by a
      random seed.
    seed2: An optional `int`. Defaults to `0`.
      An second seed to avoid seed collision.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `image`.
    3-D of shape `[crop_height, crop_width, channels].`
  """
  return _op_def_lib.apply_op("RandomCrop", image=image, size=size, seed=seed,
                              seed2=seed2, name=name)


def resize_area(images, size, align_corners=None, name=None):
  r"""Resize `images` to `size` using area interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale input by (new_height - 1) / (height - 1), which
      exactly aligns the 4 corners of images and resized images. If false, rescale
      by new_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeArea", images=images, size=size,
                              align_corners=align_corners, name=name)


def resize_bicubic(images, size, align_corners=None, name=None):
  r"""Resize `images` to `size` using bicubic interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale input by (new_height - 1) / (height - 1), which
      exactly aligns the 4 corners of images and resized images. If false, rescale
      by new_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeBicubic", images=images, size=size,
                              align_corners=align_corners, name=name)


def resize_bilinear(images, size, align_corners=None, name=None):
  r"""Resize `images` to `size` using bilinear interpolation.

  Input images can be of different types but output images are always float.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale input by (new_height - 1) / (height - 1), which
      exactly aligns the 4 corners of images and resized images. If false, rescale
      by new_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeBilinear", images=images, size=size,
                              align_corners=align_corners, name=name)


def _resize_bilinear_grad(grads, original_image, align_corners=None,
                          name=None):
  r"""Computes the gradient of bilinear interpolation.

  Args:
    grads: A `Tensor` of type `float32`.
      4-D with shape `[batch, height, width, channels]`.
    original_image: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      4-D with shape `[batch, orig_height, orig_width, channels]`,
      The image tensor that was resized.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale grads by (orig_height - 1) / (height - 1), which
      exactly aligns the 4 corners of grads and original_image. If false, rescale by
      orig_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `original_image`.
    4-D with shape `[batch, orig_height, orig_width, channels]`.
    Gradients with respect to the input image. Input image must have been
    float or double.
  """
  return _op_def_lib.apply_op("ResizeBilinearGrad", grads=grads,
                              original_image=original_image,
                              align_corners=align_corners, name=name)


def resize_nearest_neighbor(images, size, align_corners=None, name=None):
  r"""Resize `images` to `size` using nearest neighbor interpolation.

  Args:
    images: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `new_height, new_width`.  The
      new size for the images.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale input by (new_height - 1) / (height - 1), which
      exactly aligns the 4 corners of images and resized images. If false, rescale
      by new_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `images`. 4-D with shape
    `[batch, new_height, new_width, channels]`.
  """
  return _op_def_lib.apply_op("ResizeNearestNeighbor", images=images,
                              size=size, align_corners=align_corners,
                              name=name)


def _resize_nearest_neighbor_grad(grads, size, align_corners=None, name=None):
  r"""Computes the gradient of nearest neighbor interpolation.

  Args:
    grads: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int32`, `float32`, `float64`.
      4-D with shape `[batch, height, width, channels]`.
    size:  A 1-D int32 Tensor of 2 elements: `orig_height, orig_width`. The
      original input size.
    align_corners: An optional `bool`. Defaults to `False`.
      If true, rescale grads by (orig_height - 1) / (height - 1), which
      exactly aligns the 4 corners of grads and original_image. If false, rescale by
      orig_height / height. Treat similarly the width dimension.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `grads`.
    4-D with shape `[batch, orig_height, orig_width, channels]`. Gradients
    with respect to the input image.
  """
  return _op_def_lib.apply_op("ResizeNearestNeighborGrad", grads=grads,
                              size=size, align_corners=align_corners,
                              name=name)


def sample_distorted_bounding_box(image_size, bounding_boxes, seed=None,
                                  seed2=None, min_object_covered=None,
                                  aspect_ratio_range=None, area_range=None,
                                  max_attempts=None,
                                  use_image_if_no_bounding_boxes=None,
                                  name=None):
  r"""Generate a single randomly distorted bounding box for an image.

  Bounding box annotations are often supplied in addition to ground-truth labels
  in image recognition or object localization tasks. A common technique for
  training such a system is to randomly distort an image while preserving
  its content, i.e. *data augmentation*. This Op outputs a randomly distorted
  localization of an object, i.e. bounding box, given an `image_size`,
  `bounding_boxes` and a series of constraints.

  The output of this Op is a single bounding box that may be used to crop the
  original image. The output is returned as 3 tensors: `begin`, `size` and
  `bboxes`. The first 2 tensors can be fed directly into `tf.slice` to crop the
  image. The latter may be supplied to `tf.image.draw_bounding_box` to visualize
  what the bounding box looks like.

  Bounding boxes are supplied and returned as `[y_min, x_min, y_max, x_max]`. The
  bounding box coordinates are floats in `[0.0, 1.0]` relative to the width and
  height of the underlying image.

  For example,

      # Generate a single distorted bounding box.
      begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
          tf.shape(image),
          bounding_boxes=bounding_boxes)

      # Draw the bounding box in an image summary.
      image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                    bbox_for_draw)
      tf.image_summary('images_with_box', image_with_box)

      # Employ the bounding box to distort the image.
      distorted_image = tf.slice(image, begin, size)

  Note that if no bounding box information is available, setting
  `use_image_if_no_bounding_boxes = true` will assume there is a single implicit
  bounding box covering the whole image. If `use_image_if_no_bounding_boxes` is
  false and no bounding boxes are supplied, an error is raised.

  Args:
    image_size: A `Tensor`. Must be one of the following types: `uint8`, `int8`, `int16`, `int32`, `int64`.
      1-D, containing `[height, width, channels]`.
    bounding_boxes: A `Tensor` of type `float32`.
      3-D with shape `[batch, N, 4]` describing the N bounding boxes
      associated with the image.
    seed: An optional `int`. Defaults to `0`.
      If either `seed` or `seed2` are set to non-zero, the random number
      generator is seeded by the given `seed`.  Otherwise, it is seeded by a random
      seed.
    seed2: An optional `int`. Defaults to `0`.
      A second seed to avoid seed collision.
    min_object_covered: An optional `float`. Defaults to `0.1`.
      The cropped area of the image must contain at least this
      fraction of any bounding box supplied.
    aspect_ratio_range: An optional list of `floats`. Defaults to `[0.75, 1.33]`.
      The cropped area of the image must have an aspect ratio =
      width / height within this range.
    area_range: An optional list of `floats`. Defaults to `[0.05, 1]`.
      The cropped area of the image must contain a fraction of the
      supplied image within in this range.
    max_attempts: An optional `int`. Defaults to `100`.
      Number of attempts at generating a cropped region of the image
      of the specified constraints. After `max_attempts` failures, return the entire
      image.
    use_image_if_no_bounding_boxes: An optional `bool`. Defaults to `False`.
      Controls behavior if no bounding boxes supplied.
      If true, assume an implicit bounding box covering the whole input. If false,
      raise an error.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (begin, size, bboxes).
    begin: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[offset_height, offset_width, 0]`. Provide as input to
      `tf.slice`.
    size: A `Tensor`. Has the same type as `image_size`. 1-D, containing `[target_height, target_width, -1]`. Provide as input to
      `tf.slice`.
    bboxes: A `Tensor` of type `float32`. 3-D with shape `[1, 1, 4]` containing the distorted bounding box.
      Provide as input to `tf.image.draw_bounding_boxes`.
  """
  return _op_def_lib.apply_op("SampleDistortedBoundingBox",
                              image_size=image_size,
                              bounding_boxes=bounding_boxes, seed=seed,
                              seed2=seed2,
                              min_object_covered=min_object_covered,
                              aspect_ratio_range=aspect_ratio_range,
                              area_range=area_range,
                              max_attempts=max_attempts,
                              use_image_if_no_bounding_boxes=use_image_if_no_bounding_boxes,
                              name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "AdjustContrast"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "contrast_factor"
    type: DT_FLOAT
  }
  input_arg {
    name: "min_value"
    type: DT_FLOAT
  }
  input_arg {
    name: "max_value"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "AdjustContrastv2"
  input_arg {
    name: "images"
    type: DT_FLOAT
  }
  input_arg {
    name: "contrast_factor"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "DecodeJpeg"
  input_arg {
    name: "contents"
    type: DT_STRING
  }
  output_arg {
    name: "image"
    type: DT_UINT8
  }
  attr {
    name: "channels"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "ratio"
    type: "int"
    default_value {
      i: 1
    }
  }
  attr {
    name: "fancy_upscaling"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "try_recover_truncated"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "acceptable_fraction"
    type: "float"
    default_value {
      f: 1
    }
  }
}
op {
  name: "DecodePng"
  input_arg {
    name: "contents"
    type: DT_STRING
  }
  output_arg {
    name: "image"
    type_attr: "dtype"
  }
  attr {
    name: "channels"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "dtype"
    type: "type"
    default_value {
      type: DT_UINT8
    }
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_UINT16
      }
    }
  }
}
op {
  name: "DrawBoundingBoxes"
  input_arg {
    name: "images"
    type: DT_FLOAT
  }
  input_arg {
    name: "boxes"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "EncodeJpeg"
  input_arg {
    name: "image"
    type: DT_UINT8
  }
  output_arg {
    name: "contents"
    type: DT_STRING
  }
  attr {
    name: "format"
    type: "string"
    default_value {
      s: ""
    }
    allowed_values {
      list {
        s: ""
        s: "grayscale"
        s: "rgb"
      }
    }
  }
  attr {
    name: "quality"
    type: "int"
    default_value {
      i: 95
    }
  }
  attr {
    name: "progressive"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "optimize_size"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "chroma_downsampling"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "density_unit"
    type: "string"
    default_value {
      s: "in"
    }
    allowed_values {
      list {
        s: "in"
        s: "cm"
      }
    }
  }
  attr {
    name: "x_density"
    type: "int"
    default_value {
      i: 300
    }
  }
  attr {
    name: "y_density"
    type: "int"
    default_value {
      i: 300
    }
  }
  attr {
    name: "xmp_metadata"
    type: "string"
    default_value {
      s: ""
    }
  }
}
op {
  name: "EncodePng"
  input_arg {
    name: "image"
    type_attr: "T"
  }
  output_arg {
    name: "contents"
    type: DT_STRING
  }
  attr {
    name: "compression"
    type: "int"
    default_value {
      i: -1
    }
  }
  attr {
    name: "T"
    type: "type"
    default_value {
      type: DT_UINT8
    }
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_UINT16
      }
    }
  }
}
op {
  name: "HSVToRGB"
  input_arg {
    name: "images"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "RGBToHSV"
  input_arg {
    name: "images"
    type: DT_FLOAT
  }
  output_arg {
    name: "output"
    type: DT_FLOAT
  }
}
op {
  name: "RandomCrop"
  input_arg {
    name: "image"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT64
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  is_stateful: true
}
op {
  name: "ResizeArea"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResizeBicubic"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResizeBilinear"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResizeBilinearGrad"
  input_arg {
    name: "grads"
    type: DT_FLOAT
  }
  input_arg {
    name: "original_image"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResizeNearestNeighbor"
  input_arg {
    name: "images"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "resized_images"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "ResizeNearestNeighborGrad"
  input_arg {
    name: "grads"
    type_attr: "T"
  }
  input_arg {
    name: "size"
    type: DT_INT32
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT32
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "align_corners"
    type: "bool"
    default_value {
      b: false
    }
  }
}
op {
  name: "SampleDistortedBoundingBox"
  input_arg {
    name: "image_size"
    type_attr: "T"
  }
  input_arg {
    name: "bounding_boxes"
    type: DT_FLOAT
  }
  output_arg {
    name: "begin"
    type_attr: "T"
  }
  output_arg {
    name: "size"
    type_attr: "T"
  }
  output_arg {
    name: "bboxes"
    type: DT_FLOAT
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_UINT8
        type: DT_INT8
        type: DT_INT16
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
  attr {
    name: "seed"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "seed2"
    type: "int"
    default_value {
      i: 0
    }
  }
  attr {
    name: "min_object_covered"
    type: "float"
    default_value {
      f: 0.1
    }
  }
  attr {
    name: "aspect_ratio_range"
    type: "list(float)"
    default_value {
      list {
        f: 0.75
        f: 1.33
      }
    }
  }
  attr {
    name: "area_range"
    type: "list(float)"
    default_value {
      list {
        f: 0.05
        f: 1
      }
    }
  }
  attr {
    name: "max_attempts"
    type: "int"
    default_value {
      i: 100
    }
  }
  attr {
    name: "use_image_if_no_bounding_boxes"
    type: "bool"
    default_value {
      b: false
    }
  }
  is_stateful: true
}
"""


_op_def_lib = _InitOpDefLibrary()
