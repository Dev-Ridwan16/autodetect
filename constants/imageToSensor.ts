import * as jpeg from 'jpeg-js'
import * as tf from '@tensorflow/tfjs'

export default async function imageToTensor(
  base64: string
): Promise<tf.Tensor4D> {
  const rawImageData = jpeg.decode(Buffer.from(base64, 'base64'), {
    useTArray: true,
  })
  const { width, height, data } = rawImageData

  // Normalize RGB channels
  const buffer = new Float32Array(width * height * 3)
  let offset = 0

  for (let i = 0; i < data.length; i += 4) {
    buffer[offset++] = data[i] / 255 // R
    buffer[offset++] = data[i + 1] / 255 // G
    buffer[offset++] = data[i + 2] / 255 // B
  }

  const imageTensor = tf.tensor3d(buffer, [height, width, 3])

  // Resize to 224x224 and expand dims to [1, 224, 224, 3]
  const resized = tf.image.resizeBilinear(imageTensor, [224, 224])
  const batched = resized.expandDims(0)

  return batched as tf.Tensor4D
}
