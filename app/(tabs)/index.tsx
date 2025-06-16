import ParallaxScrollView from '@/components/ParallaxScrollView'
import '@tensorflow/tfjs-react-native'
import * as tf from '@tensorflow/tfjs'
import { Platform } from 'react-native'
import * as ImagePicker from 'expo-image-picker'

import { bundleResourceIO } from '@tensorflow/tfjs-react-native'
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native'
import { useEffect, useRef, useState, useCallback } from 'react'
import { ThemedText } from '@/components/ThemedText'
import {
  Camera,
  CameraType,
  CameraView,
  useCameraPermissions,
} from 'expo-camera'
import imageToTensor from '@/constants/imageToSensor'

// Initialize TensorFlow platform outside component to avoid re-initialization
let tfInitialized = false
let modelCache: tf.LayersModel | null = null

const initializeTensorFlow = async () => {
  if (tfInitialized) return true

  try {
    // Platform-specific initialization
    if (Platform.OS === 'ios') {
      await tf.ready()
    } else {
      // Android optimization
      await tf.ready()
    }

    tfInitialized = true
    console.log('TensorFlow initialized successfully')
    return true
  } catch (error) {
    console.error('TensorFlow initialization failed:', error)
    return false
  }
}

const loadModelAsync = async (onProgress?: (progress: number) => void) => {
  if (modelCache) return modelCache

  try {
    onProgress?.(10) // Starting model load

    const modelJson = require('../../assets/model.json')
    const modelWeight = require('../../assets/weights.bin')
    console.log('Loaded model.json:', modelJson)

    onProgress?.(30) // Assets loaded

    const loadedModel = await tf.loadLayersModel(
      bundleResourceIO(modelJson, modelWeight)
    )

    onProgress?.(80) // Model loaded

    // Warm up the model with a dummy prediction to improve first inference speed
    const dummyInput = tf.zeros([1, 224, 224, 3]) // Adjust dimensions to match your model
    await loadedModel.predict(dummyInput)
    dummyInput.dispose()

    onProgress?.(100) // Model ready

    modelCache = loadedModel
    console.log('Model loaded and warmed up successfully')
    return loadedModel
  } catch (error) {
    console.error('Model loading failed:', error)
    throw error
  }
}

export default function HomeScreen() {
  const [permission, requestPermission] = useCameraPermissions()
  const [loadingProgress, setLoadingProgress] = useState(0)
  const [loadingStage, setLoadingStage] = useState('Initializing...')
  const [isReady, setIsReady] = useState(false)
  const [model, setModel] = useState<tf.LayersModel | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [facing, setFacing] = useState<CameraType>('back')
  const [analyzing, setAnalyzing] = useState<boolean>(false)
  const [predictionResult, setPredictionResult] = useState<
    { label: string; confidence: string }[] | null
  >(null)

  const classNames = ['Clean', 'Dirty', 'Invalid']

  const cameraRef = useRef<CameraView>(null)

  const initializeApp = useCallback(async () => {
    try {
      setLoadingStage('Initializing TensorFlow...')
      setLoadingProgress(5)

      const tfReady = await initializeTensorFlow()
      if (!tfReady) {
        throw new Error('Failed to initialize TensorFlow')
      }

      setLoadingProgress(20)
      setLoadingStage('Loading AI Model...')

      const loadedModel = await loadModelAsync((progress) => {
        setLoadingProgress(20 + progress * 0.8) // Scale progress from 20-100
        if (progress === 30) setLoadingStage('Processing model assets...')
        if (progress === 80) setLoadingStage('Warming up model...')
        if (progress === 100) setLoadingStage('Ready!')
      })

      setModel(loadedModel)
      setIsReady(true)
      setLoadingProgress(100)
    } catch (error: any) {
      console.error('Initialization error:', error)
      setError(error.message || 'Failed to initialize app')
    }
  }, [])

  useEffect(() => {
    initializeApp()
  }, [initializeApp])

  // Request media library permissions
  const requestMediaLibraryPermission = async () => {
    const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync()
    if (status !== 'granted') {
      alert('Sorry, we need camera roll permissions to upload images!')
      return false
    }
    return true
  }

  const formatBase64Image = (base64: string, mimeType?: string) => {
    // If the base64 already has a data URL prefix, return as is
    if (base64.startsWith('data:')) {
      return base64
    }

    // Otherwise, add the data URL prefix
    const mime = mimeType || 'image/jpeg'
    return `data:${mime};base64,${base64}`
  }

  // Handle image upload from gallery
  const handleImageUpload = async () => {
    const hasPermission = await requestMediaLibraryPermission()
    if (!hasPermission) return

    try {
      const result = await ImagePicker.launchImageLibraryAsync({
        mediaTypes: ImagePicker.MediaTypeOptions.Images,
        allowsEditing: true,
        aspect: [1, 1],
        quality: 1,
        base64: true,
      })

      if (!result.canceled && result.assets[0]) {
        const selectedImage = result.assets[0]
        console.log('Image selected:', selectedImage.uri)

        if (!selectedImage.base64) {
          throw new Error('Base64 not available')
        }

        const formattedBase64 = formatBase64Image(
          selectedImage.base64,
          selectedImage.mimeType || 'image/jpeg'
        )

        console.log(
          'Formatted base64 prefix:',
          formattedBase64.substring(0, 50)
        )

        setAnalyzing(true)
        await processImage(selectedImage.base64)
      } else {
        setAnalyzing(false)
      }
    } catch (error) {
      console.log('Error uploading image:', error)
      setPredictionResult([
        { label: 'Error: Failed to upload image.', confidence: 'null' },
      ])
      setAnalyzing(false)
    }
  }

  // Process image with AI model
  const processImage = async (base64Image: string) => {
    if (!model) {
      setPredictionResult([
        { label: 'Error: Model not loaded.', confidence: 'null' },
      ])
      setAnalyzing(false)
      return
    }

    try {
      const imageTensor = await imageToTensor(base64Image)
      const prediction = model.predict(imageTensor) as tf.Tensor
      const result = await prediction.data()
      const predictedClass = result.indexOf(Math.max(...result))

      console.log('Prediction result:', result)
      const resultWithAnalysis = classNames.map((label, index) => ({
        label,
        confidence: (result[index] * 100).toFixed(2) + '%',
      }))

      imageTensor.dispose()
      prediction.dispose()

      setPredictionResult(resultWithAnalysis)
    } catch (error) {
      console.log('Error processing image:', error)
      setPredictionResult([
        { label: 'Error: Failed to process image.', confidence: 'null' },
      ])
    } finally {
      setAnalyzing(false)
    }
  }

  const handleCapture = async () => {
    setAnalyzing(true)
    if (cameraRef.current && model) {
      try {
        const photo = await cameraRef.current.takePictureAsync({
          quality: 1,
          base64: true,
        })
        console.log('Photo captured:', photo.uri)

        if (!photo.base64) throw new Error('Base64 not available')

        await processImage(photo.base64)
      } catch (error) {
        console.log('Error capturing photo:', error)
        setPredictionResult([
          { label: 'Error: Failed to capture photo.', confidence: 'null' },
        ])
        setAnalyzing(false)
      }
    }
  }

  function toggleCameraFacing() {
    setFacing((current) => (current === 'back' ? 'front' : 'back'))
  }

  // Error state
  if (error) {
    return (
      <View style={styles.centerContainer}>
        <ThemedText style={styles.errorText}>❌ {error}</ThemedText>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={() => {
            setError(null)
            setLoadingProgress(0)
            setIsReady(false)
            initializeApp()
          }}
        >
          <Text style={styles.buttonText}>Retry</Text>
        </TouchableOpacity>
      </View>
    )
  }

  // Permission loading
  if (!permission) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size='large' color='#4caf50' />
        <ThemedText>Loading camera permissions...</ThemedText>
      </View>
    )
  }

  // Permission denied
  if (!permission.granted) {
    return (
      <View style={styles.centerContainer}>
        <ThemedText style={styles.permissionText}>
          📷 Camera access is required for AI analysis
        </ThemedText>
        <TouchableOpacity style={styles.button} onPress={requestPermission}>
          <Text style={styles.buttonText}>Grant Camera Permission</Text>
        </TouchableOpacity>
      </View>
    )
  }

  // Loading state with progress
  if (!isReady || !model) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size='large' color='#4caf50' />
        <ThemedText style={styles.loadingTitle}>Setting up AI...</ThemedText>
        <ThemedText style={styles.loadingStage}>{loadingStage}</ThemedText>
        <View style={styles.progressContainer}>
          <View
            style={[styles.progressBar, { width: `${loadingProgress}%` }]}
          />
        </View>
        <ThemedText style={styles.progressText}>
          {Math.round(loadingProgress)}%
        </ThemedText>
      </View>
    )
  }

  return (
    <ParallaxScrollView
      headerBackgroundColor={{ light: '#A1CEDC', dark: '#1D3D47' }}
    >
      <CameraView style={styles.camera} facing={facing} ref={cameraRef} />

      <View style={styles.buttonContainer}>
        <TouchableOpacity
          style={styles.secondaryButton}
          onPress={toggleCameraFacing}
        >
          <ThemedText style={styles.buttonText}>📷 Flip Camera</ThemedText>
        </TouchableOpacity>
      </View>

      <View style={styles.overlay}>
        <ThemedText style={styles.readyText}>🤖 AI Ready</ThemedText>
      </View>

      <View style={styles.buttonContainer}>
        <TouchableOpacity style={styles.captureButton} onPress={handleCapture}>
          <ThemedText style={styles.buttonText}>📸 Take Photo</ThemedText>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.uploadButton}
          onPress={handleImageUpload}
        >
          <ThemedText style={styles.buttonText}>🖼️ Upload Image</ThemedText>
        </TouchableOpacity>
      </View>

      {(analyzing || predictionResult) && (
        <View style={styles.analyzingModal}>
          {analyzing ? (
            <>
              <ActivityIndicator size='small' color='#4caf50' />
              <ThemedText style={styles.analyzingText}>Analyzing...</ThemedText>
            </>
          ) : (
            <>
              {/* 🧪 Prediction: {predictionResult} */}
              {Array.isArray(predictionResult) &&
                predictionResult.map((res, idx) => (
                  <ThemedText style={styles.resultText}>
                    {res.label}: {res.confidence}
                  </ThemedText>
                ))}

              <TouchableOpacity
                style={styles.closeButton}
                onPress={() => setPredictionResult(null)}
              >
                <ThemedText style={styles.closeButtonText}>Close</ThemedText>
              </TouchableOpacity>
            </>
          )}
        </View>
      )}
    </ParallaxScrollView>
  )
}

export const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    width: '100%',
    height: 300,
    borderWidth: 1,
    marginTop: 50,
  },
  overlay: {
    position: 'absolute',
    top: 81,
    left: 34,
    backgroundColor: 'rgba(0,0,0,0.7)',
    padding: 10,
    borderRadius: 8,
  },
  readyText: {
    color: '#4caf50',
    fontSize: 16,
    fontWeight: 'bold',
  },
  buttonContainer: {
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    marginVertical: 10,
    paddingHorizontal: 20,
  },
  captureButton: {
    backgroundColor: '#4caf50',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    flex: 1,
    marginRight: 10,
  },
  uploadButton: {
    backgroundColor: '#2196f3',
    paddingHorizontal: 20,
    paddingVertical: 15,
    borderRadius: 25,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 3.84,
    flex: 1,
    marginLeft: 10,
  },
  secondaryButton: {
    backgroundColor: '#ff9800',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 20,
    elevation: 3,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.22,
    shadowRadius: 2.22,
  },
  button: {
    backgroundColor: '#4caf50',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  retryButton: {
    backgroundColor: '#f44336',
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
    marginTop: 20,
  },
  buttonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  loadingTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginTop: 20,
    marginBottom: 10,
  },
  loadingStage: {
    fontSize: 14,
    opacity: 0.7,
    marginBottom: 20,
  },
  progressContainer: {
    width: 200,
    height: 6,
    backgroundColor: '#e0e0e0',
    borderRadius: 3,
    overflow: 'hidden',
    marginBottom: 10,
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#4caf50',
    borderRadius: 3,
  },
  progressText: {
    fontSize: 12,
    opacity: 0.7,
  },
  permissionText: {
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 20,
  },
  errorText: {
    color: '#f44336',
    textAlign: 'center',
    fontSize: 16,
    marginBottom: 20,
  },
  text: {
    fontSize: 24,
    fontWeight: 'bold',
    color: 'white',
  },
  analyzingModal: {
    position: 'absolute',
    width: '100%',
    height: 200,
    top: '75%',
    left: '5%',
    backgroundColor: '#f5f5f5',
    borderColor: '#ddd',
    borderWidth: 1,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 10,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.34,
    shadowRadius: 6.27,
  },
  analyzingText: {
    color: '#4caf50',
    fontSize: 16,
    marginTop: 10,
  },
  resultText: {
    color: '#2b2b2b',
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  closeButton: {
    backgroundColor: '#f44336',
    paddingHorizontal: 20,
    paddingVertical: 8,
    borderRadius: 15,
  },
  closeButtonText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
})
